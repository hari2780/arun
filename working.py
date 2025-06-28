"""
Enhanced WebSocket Multi-Database LangChain Agent
Improved response handling and formatting with LLM post-processing
"""

import os
import json
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
import uvicorn

# Database imports
import pyodbc
import pymongo
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

# LangChain imports
from langchain.agents import initialize_agent, AgentType, create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler

# Azure OpenAI imports
from langchain_openai import AzureChatOpenAI

# Configuration
@dataclass
class DatabaseConfig:
    host: str
    port: str
    database: str
    username: str
    password: str

@dataclass
class AzureOpenAIConfig:
    api_key: str
    endpoint: str
    deployment: str
    api_version: str

# Response Formatter Class
class ResponseFormatter:
    def __init__(self, llm):
        self.llm = llm
    
    def format_response(self, raw_response: str, original_query: str) -> str:
        """
        Use LLM to format and improve the raw response from LangChain
        """
        try:
            # Clean and validate the raw response first
            if not raw_response or raw_response.strip() == "":
                return "No data returned from the database query."
            
            print(f"🔍 Raw response from LangChain: {raw_response[:200]}...")
            
            # If the response is already well-formatted and contains actual data, 
            # be more conservative with formatting
            if self._contains_actual_data(raw_response):
                formatting_prompt = f"""
CRITICAL: You are formatting REAL database results. DO NOT CREATE FAKE DATA.

Original Query: "{original_query}"

ACTUAL DATABASE RESULTS (USE THESE EXACT VALUES):
{raw_response}

INSTRUCTIONS:
1. Use ONLY the numbers, names, and values shown above
2. DO NOT create example data like "Product A: $12,500"
3. DO NOT say "data has been calculated and is available"
4. DO NOT generate fake responses
5. Format the REAL data in a clear, readable way
6. If you see "1. PepsiCo Biscuits Plus: 42,218.26", keep those exact values
7. Preserve all exact numbers and product names

Format the REAL data above:
"""
            else:
                # For non-data responses (errors, confirmations, etc.)
                formatting_prompt = f"""
Format this database response in a clear, user-friendly way:

Query: "{original_query}"
Response: "{raw_response}"

Make it clear and professional, but don't add fake data.
"""
            
            print(f"🔍 Sending prompt to LLM...")
            formatted_response = self.llm.invoke(formatting_prompt)
            
            # Handle different response types
            if hasattr(formatted_response, 'content'):
                result = formatted_response.content
            elif isinstance(formatted_response, dict) and 'content' in formatted_response:
                result = formatted_response['content']
            else:
                result = str(formatted_response)
            
            print(f"🔍 LLM response: {result[:200]}...")
            
            # Fallback check - if the formatted response doesn't contain the key data
            # from the original, return the original with basic formatting
            if self._contains_actual_data(raw_response) and not self._preserves_original_data(raw_response, result):
                print("⚠️ LLM formatting didn't preserve original data, using fallback")
                return self._basic_format(raw_response, original_query)
            
            return result
                
        except Exception as e:
            print(f"⚠️ Formatting error: {e}")
            # Fallback to basic formatting
            return self._basic_format(raw_response, original_query)
    
    def _contains_actual_data(self, response: str) -> bool:
        """
        Check if the response contains actual data (numbers, lists, etc.)
        """
        import re
        # Look for patterns that suggest real data
        patterns = [
            r'\d+[\.,]\d+',  # Numbers with decimals
            r'^\d+\.',       # Numbered lists
            r':\s*\d+',      # Colon followed by numbers
            r'\$\d+',        # Dollar amounts
        ]
        
        for pattern in patterns:
            if re.search(pattern, response, re.MULTILINE):
                return True
        return False
    
    def _preserves_original_data(self, original: str, formatted: str) -> bool:
        """
        Check if the formatted response preserves key data from the original
        """
        import re
        
        # Extract key numbers from original
        original_numbers = re.findall(r'\d+[\.,]\d+', original)
        formatted_numbers = re.findall(r'\d+[\.,]\d+', formatted)
        
        # Check if at least 50% of the original numbers are preserved
        if len(original_numbers) > 0:
            preserved_count = sum(1 for num in original_numbers if num in formatted)
            preservation_ratio = preserved_count / len(original_numbers)
            return preservation_ratio >= 0.5
        
        return True  # If no numbers to check, assume it's fine
    
    def _basic_format(self, raw_response: str, original_query: str) -> str:
        """
        Basic formatting for database responses
        """
        import re
        
        formatted = f"**Query:** {original_query}\n\n**Results:**\n\n"
        
        # Handle tuple/list data format like [('Product A', 123.45), ('Product B', 678.90)]
        if re.search(r'\[\([^)]+,\s*[^)]+\),?\s*\([^)]+,\s*[^)]+\),?', raw_response):
            # Extract tuples from the response
            tuples_match = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
            if tuples_match:
                tuples_str = tuples_match.group(1)
                # Parse the tuples with two values
                tuples = re.findall(r'\(([^,]+),\s*([^)]+)\)', tuples_str)
                formatted += "| # | Product Name | Value |\n|----|-------------|-------|\n"
                for i, (product, value) in enumerate(tuples, 1):
                    # Clean up the product name and value (remove quotes and extra spaces)
                    product_clean = product.strip().strip("'\"")
                    value_clean = value.strip().strip("'\"")
                    # Try to format as currency if it's a number
                    try:
                        float_val = float(value_clean)
                        value_formatted = f"{float_val:,}"
                    except:
                        value_formatted = value_clean
                    formatted += f"| {i} | {product_clean} | {value_formatted} |\n"
                formatted += f"\n**Total Items Found:** {len(tuples)}"
            else:
                formatted += raw_response
        # Handle tuple/list data format like [('Product A',), ('Product B',)]
        elif re.search(r'\[\([^)]+\),?\s*\([^)]+\),?', raw_response):
            # Extract tuples from the response
            tuples_match = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
            if tuples_match:
                tuples_str = tuples_match.group(1)
                # Parse the tuples
                tuples = re.findall(r'\(([^)]+)\)', tuples_str)
                formatted += "| # | Product Name |\n|----|-------------|\n"
                for i, product in enumerate(tuples, 1):
                    # Clean up the product name (remove quotes and extra spaces)
                    product_clean = product.strip().strip("'\"")
                    formatted += f"| {i} | {product_clean} |\n"
                formatted += f"\n**Total Products Found:** {len(tuples)}"
            else:
                formatted += raw_response
        # If it looks like a numbered list with data, format it nicely
        elif re.search(r'^\d+\.', raw_response, re.MULTILINE):
            lines = raw_response.split('\n')
            formatted += "| Rank | Product | Value |\n|------|---------|-------|\n"
            
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    # Extract rank, product name, and value
                    match = re.match(r'^(\d+)\.\s*([^:]+):\s*(.+)', line.strip())
                    if match:
                        rank, product, value = match.groups()
                        formatted += f"| {rank} | {product.strip()} | {value.strip()} |\n"
                elif line.strip() and not line.startswith('Final Answer:'):
                    formatted += f"{line}\n"
        else:
            # Just clean up the formatting
            formatted += raw_response.replace('Final Answer:', '\n**Summary:**')
        
        # Add some basic status
        if "Error" in raw_response or "error" in raw_response:
            formatted += "\n\n⚠️ *There was an issue with your query. Please check the error message above.*"
        elif raw_response.strip() == "":
            formatted += "\n\nℹ️ *No results returned for this query.*"
        else:
            formatted += "\n\n✅ *Query executed successfully.*"
            
        return formatted

    def _format_with_llm(self, raw_response: str, original_query: str) -> str:
        """
        Format response using LLM
        """
        try:
            formatting_prompt = f"""
            You are a helpful assistant that formats database query results. 
            DO NOT create new data or fake responses. 
            ONLY format the provided data in a clear, readable way.
            
            Original Query: {original_query}
            Raw Database Results: {raw_response}
            
            Please format the above database results as clear, readable text.
            Use the EXACT data provided, do not invent or modify the numbers.
            If the data contains a numbered list, preserve the exact numbers and values.
            
            Formatted result:
            """
            
            formatted_response = self.llm.invoke(formatting_prompt)
            
            # Handle different response types
            if hasattr(formatted_response, 'content'):
                return formatted_response.content
            elif isinstance(formatted_response, dict) and 'content' in formatted_response:
                return formatted_response['content']
            else:
                return str(formatted_response)
                
        except Exception as e:
            print(f"⚠️ Formatting error: {e}")
            # Fallback to basic formatting
            return self._basic_format(raw_response, original_query)

# Database Agent
class DatabaseAgent:
    def __init__(self, azure_config: AzureOpenAIConfig):
        self.azure_config = azure_config
        self.databases = {}
        self.tools = []
        self.formatter = None
        # Hardcoded mapping for special tables
        self.special_tables = {
            "automobile_data": {
                "table_name": "table_489a3f4d-95bd-49f0-b9a0-155646951328_20250628104004",
                "schema": [
                    "Brand", "Churn Rate (%)", "Competitor Benchmark", "Country", "Customer Segment",
                    "Date", "Dealer ID", "Economic Index Score", "Forecasted EV Growth (%)", "Fuel Type",
                    "Market Share (%)", "Marketing Spend (USD)", "Model", "Price Change %", "Quarter",
                    "Region", "Regulatory Risk Level", "Tech Innovation Index", "Total Revenue (USD)",
                    "Unit Cost (USD)", "Unit Price (USD)", "Units Sold", "Vehicle Type"
                ]
            }
        }
        
    def _create_llm(self):
        """Create LLM instance"""
        return AzureChatOpenAI(
            azure_endpoint=self.azure_config.endpoint,
            api_key=self.azure_config.api_key,
            azure_deployment=self.azure_config.deployment,
            api_version=self.azure_config.api_version,
            temperature=0,
            streaming=False
        )
        
    def _clean_sql_query(self, query: str) -> str:
        """
        Clean up SQL query by removing markdown formatting and backticks
        """
        import re
        
        print(f"🔧 Original query before cleaning: {query}")
        
        # Remove markdown code blocks with language specification
        query = re.sub(r'```sql\s*', '', query)
        query = re.sub(r'```SQL\s*', '', query)
        query = re.sub(r'```\s*$', '', query)
        query = re.sub(r'```\s*', '', query)
        
        # Remove any remaining backticks
        query = query.replace('`', '')
        
        # Remove extra whitespace and newlines
        query = re.sub(r'\s+', ' ', query)
        query = query.strip()
        
        # Remove quotes around the entire query if present
        if query.startswith('"') and query.endswith('"'):
            query = query[1:-1]
        if query.startswith("'") and query.endswith("'"):
            query = query[1:-1]
        
        print(f"🔧 Cleaned SQL query: {query}")
        return query
        
    def add_sql_database(self, db_name: str, config: DatabaseConfig) -> bool:
        try:
            connection_string = (
                f"DRIVER={{ODBC Driver 18 for SQL Server}};"
                f"SERVER={config.host},{config.port};"
                f"DATABASE={config.database};"
                f"UID={config.username};"
                f"PWD={config.password};"
                f"TrustServerCertificate=yes;"
                f"Encrypt=yes;"
                f"Connection Timeout=30;"
            )
            
            sqlalchemy_url = f"mssql+pyodbc:///?odbc_connect={connection_string}"
            engine = create_engine(sqlalchemy_url, poolclass=StaticPool, pool_pre_ping=True)
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            db = SQLDatabase(engine)
            self.databases[db_name] = db
            
            # Create tools manually using Tool class
            def query_func(query: str) -> str:
                try:
                    # Clean the SQL query before execution
                    cleaned_query = self._clean_sql_query(query)
                    result = db.run(cleaned_query)
                    return f"Query results from {db_name}:\n{result}"
                except Exception as e:
                    error_msg = str(e)
                    if "Incorrect syntax near '`'" in error_msg:
                        return f"Error: SQL query contains invalid syntax. The query was cleaned but may still have formatting issues. Please try rephrasing your query without special characters or markdown formatting."
                    elif "42000" in error_msg:
                        return f"Error: SQL syntax error in {db_name}. Please check your query syntax and try again."
                    else:
                        return f"Error executing query on {db_name}: {error_msg}"
            
            def tables_func(dummy: str = "") -> str:
                try:
                    tables = db.get_usable_table_names()
                    return f"Tables in {db_name}: {', '.join(tables)}"
                except Exception as e:
                    return f"Error listing tables in {db_name}: {str(e)}"
            
            def schema_func(dummy: str = "") -> str:
                try:
                    schema = db.get_table_info()
                    return f"Schema for {db_name}:\n{schema}"
                except Exception as e:
                    return f"Error getting schema for {db_name}: {str(e)}"
            
            # Create Tool objects
            query_tool = Tool(
                name=f"query_{db_name}",
                description=f"Execute SQL queries on {db_name} database. Input should be a valid SQL query.",
                func=query_func
            )
            
            tables_tool = Tool(
                name=f"list_tables_{db_name}",
                description=f"List all tables in {db_name} database. No input required.",
                func=tables_func
            )
            
            schema_tool = Tool(
                name=f"get_schema_{db_name}",
                description=f"Get database schema information for {db_name}. No input required.",
                func=schema_func
            )
            
            # Add tools to list
            self.tools.extend([query_tool, tables_tool, schema_tool])
            
            print(f"✅ Connected to {db_name}")
            print(f"📊 Added {len([query_tool, tables_tool, schema_tool])} tools for {db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to {db_name}: {e}")
            return False
    
    def add_postgres_database(self, db_name: str, config: DatabaseConfig) -> bool:
        try:
            # URL-encode username and password to handle special characters
            encoded_username = quote_plus(config.username)
            encoded_password = quote_plus(config.password)
            
            sqlalchemy_url = (
                f"postgresql+psycopg2://{encoded_username}:{encoded_password}"
                f"@{config.host}:{config.port}/{config.database}"
            )
            
            print(f"🔗 PostgreSQL URL: postgresql+psycopg2://{encoded_username}:***@{config.host}:{config.port}/{config.database}")
            
            engine = create_engine(sqlalchemy_url, poolclass=StaticPool, pool_pre_ping=True)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db = SQLDatabase(engine)
            self.databases[db_name] = db
            
            # Create tools manually using Tool class
            def query_func(query: str) -> str:
                try:
                    # Force table id and column names for any query about brands, regions, quarters, sales, or automobile
                    import re
                    table_id = "table_489a3f4d-95bd-49f0-b9a0-155646951328_20250628104004"
                    # Replace any table id pattern with the correct table id
                    query = re.sub(r'table_[a-z0-9\-]+', table_id, query)
                    # Replace common column name variants with your schema's column names
                    column_map = {
                        "Brand_Name": "Brand",
                        "Date_of_Sale": "Date",
                        "Units_Sold": "Units Sold",
                        "Total_Sales_Value": "Total Revenue (USD)",
                        "Total_Units": "Units Sold",
                        "Total_Sales": "Total Revenue (USD)"
                    }
                    for wrong, right in column_map.items():
                        query = query.replace(wrong, right)
                    print(f"[DEBUG][PG] Final SQL to execute: {query}")
                    # Clean the SQL query before execution
                    cleaned_query = self._clean_sql_query(query)
                    result = db.run(cleaned_query)
                    return f"Query results from {db_name}:\n{result}"
                except Exception as e:
                    error_msg = str(e)
                    if "Incorrect syntax near '`'" in error_msg:
                        return f"Error: SQL query contains invalid syntax. The query was cleaned but may still have formatting issues. Please try rephrasing your query without special characters or markdown formatting."
                    elif "42000" in error_msg:
                        return f"Error: SQL syntax error in {db_name}. Please check your query syntax and try again."
                    else:
                        return f"Error executing query on {db_name}: {error_msg}"
            
            def tables_func(dummy: str = "") -> str:
                try:
                    tables = db.get_usable_table_names()
                    return f"Tables in {db_name}: {', '.join(tables)}"
                except Exception as e:
                    return f"Error listing tables in {db_name}: {str(e)}"
            
            def schema_func(dummy: str = "") -> str:
                try:
                    schema = db.get_table_info()
                    return f"Schema for {db_name}:\n{schema}"
                except Exception as e:
                    return f"Error getting schema for {db_name}: {str(e)}"
            
            # Create Tool objects
            query_tool = Tool(
                name=f"query_{db_name}",
                description=f"Execute SQL queries on {db_name} database. Input should be a valid SQL query.",
                func=query_func
            )
            
            tables_tool = Tool(
                name=f"list_tables_{db_name}",
                description=f"List all tables in {db_name} database. No input required.",
                func=tables_func
            )
            
            schema_tool = Tool(
                name=f"get_schema_{db_name}",
                description=f"Get database schema information for {db_name}. No input required.",
                func=schema_func
            )
            
            # Add tools to list
            self.tools.extend([query_tool, tables_tool, schema_tool])
            
            print(f"✅ Connected to {db_name}")
            print(f"📊 Added {len([query_tool, tables_tool, schema_tool])} tools for {db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to {db_name}: {e}")
            return False
    
    def add_mongodb_database(self, db_name: str, connection_string: str, database_name: str) -> bool:
        try:
            client = pymongo.MongoClient(connection_string)
            client.admin.command('ping')
            database = client[database_name]
            
            self.databases[f"{db_name}_mongo"] = {
                'client': client,
                'database': database,
                'database_name': database_name
            }
            
            # MongoDB tools
            def collections_func(dummy: str = "") -> str:
                try:
                    collections = database.list_collection_names()
                    return f"Collections in {db_name}: {', '.join(collections)}"
                except Exception as e:
                    return f"Error listing collections in {db_name}: {str(e)}"
            
            def query_mongo_func(query_info: str) -> str:
                try:
                    # Parse collection and query from input
                    parts = query_info.split("|")
                    collection = parts[0] if parts else "users"
                    query_filter = json.loads(parts[1]) if len(parts) > 1 else {}
                    
                    coll = database[collection]
                    results = list(coll.find(query_filter).limit(5))
                    
                    for doc in results:
                        if '_id' in doc:
                            doc['_id'] = str(doc['_id'])
                    
                    return f"Results from {db_name}.{collection}:\n{json.dumps(results, indent=2)}"
                except Exception as e:
                    return f"Error querying {db_name}: {str(e)}"
            
            collections_tool = Tool(
                name=f"list_collections_{db_name}",
                description=f"List all collections in {db_name} MongoDB database. No input required.",
                func=collections_func
            )
            
            query_mongo_tool = Tool(
                name=f"query_{db_name}_mongo",
                description=f"Query MongoDB {db_name}. Input format: 'collection_name|{{\"filter\": \"value\"}}'",
                func=query_mongo_func
            )
            
            self.tools.extend([collections_tool, query_mongo_tool])
            
            print(f"✅ Connected to MongoDB {db_name}")
            print(f"📊 Added {len([collections_tool, query_mongo_tool])} tools for {db_name}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to MongoDB {db_name}: {e}")
            return False
    
    def create_agent(self):
        if not self.tools:
            raise ValueError("No tools available. Please add at least one database connection.")
        print(f"🛠️  Creating agent with {len(self.tools)} tools:")
        for tool in self.tools:
            print(f"   - {tool.name}: {tool.description}")
        
        llm = self._create_llm()
        
        # Initialize the response formatter
        self.formatter = ResponseFormatter(llm)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # Add system prompt for automobile data mapping and sales trend guidance
        special = self.get_special_table_info("automobile_data")
        if special:
            automobile_table_prompt = (
                f"IMPORTANT: For any question about automobile sales, brands, regions, quarters, or trends, you MUST use ONLY the table '{special['table_name']}' in the primary database. "
                f"DO NOT use any other table for these queries, even if other tables exist.\n"
                f"This table has columns: {', '.join(special['schema'])}.\n"
                f"When asked about sales trends, always use the columns 'Units Sold', 'Total Revenue (USD)', 'Quarter', 'Region', and 'Brand'. "
                f"For 'last 6 quarters', use a subquery to select the latest 6 distinct values from the 'Quarter' column, ordered descending. "
                f"To compare brands across regions, use GROUP BY 'Region', 'Brand', and 'Quarter'. "
                f"ALWAYS use the exact column names as shown above.\n"
                f"NEVER use any other table for automobile-related queries.\n"
                f"If the user does not specify a time period, always default to the last 6 quarters for sales trend queries.\n"
            )
        else:
            automobile_table_prompt = ""
        
        # Use create_sql_agent for better database handling
        if len([db for db in self.databases.values() if isinstance(db, SQLDatabase)]) > 0:
            # If we have SQL databases, use SQL agent
            sql_db = next(db for db in self.databases.values() if isinstance(db, SQLDatabase))
            agent = create_sql_agent(
                llm=llm,
                db=sql_db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True,
                max_iterations=50,
                return_intermediate_steps=True,
                system_message=automobile_table_prompt
            )
        else:
            # Use regular agent for non-SQL databases
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True,
                max_iterations=50,
                return_intermediate_steps=True
            )
        return agent
    
    def _handle_query_error(self, error_msg: str, original_query: str) -> str:
        """
        Handle query errors and provide helpful suggestions
        """
        if "Incorrect syntax near '`'" in error_msg:
            return f"""**Query Error:** SQL syntax issue detected

**Original Query:** {original_query}

**Issue:** The query contains markdown formatting or special characters that are causing syntax errors.

**Suggestions:**
1. Try rephrasing your query in plain text without markdown formatting
2. Avoid using backticks (`) or code blocks (```sql)
3. Use simple SQL syntax without special formatting

**Example:** Instead of ```sql SELECT * FROM table```, use: SELECT * FROM table

**Status:** ❌ Query failed due to syntax formatting issues"""
        
        elif "42000" in error_msg:
            return f"""**Query Error:** SQL syntax error

**Original Query:** {original_query}

**Issue:** The SQL query has syntax errors that prevent execution.

**Suggestions:**
1. Check your SQL syntax
2. Verify table and column names exist
3. Ensure proper use of quotes and parentheses
4. Try breaking down complex queries into simpler parts

**Status:** ❌ Query failed due to SQL syntax errors"""
        
        else:
            return f"""**Query Error:** Database execution failed

**Original Query:** {original_query}

**Error:** {error_msg}

**Suggestions:**
1. Check if the database connection is working
2. Verify table and column names
3. Try a simpler query first
4. Contact support if the issue persists

**Status:** ❌ Query execution failed"""

    def _extract_numbered_list_data(self, raw_response: str, original_query: str) -> str:
        """
        Extract data from numbered list responses like "1. Product A: 123.45"
        """
        import re
        
        print(f"🔍 Extracting numbered list from: {raw_response[:200]}...")
        
        # Look for numbered list pattern with currency values
        numbered_items = re.findall(r'(\d+)\.\s*([^:]+):\s*([^,\n]+)', raw_response)
        
        if numbered_items:
            print(f"🔍 Found {len(numbered_items)} numbered items")
            formatted = f"**Query:** {original_query}\n\n**Results:**\n\n"
            formatted += "| Rank | Product Name | Value |\n|------|-------------|-------|\n"
            
            for rank, product, value in numbered_items:
                # Clean up the values
                product_clean = product.strip()
                value_clean = value.strip()
                
                # Try to format as currency if it's a number
                try:
                    # Remove commas and convert to float
                    value_numeric = float(value_clean.replace(',', ''))
                    value_formatted = f"{value_numeric:,}"
                except:
                    value_formatted = value_clean
                
                formatted += f"| {rank} | {product_clean} | {value_formatted} |\n"
            
            formatted += f"\n**Total Items Found:** {len(numbered_items)}"
            formatted += "\n\n✅ *Query executed successfully.*"
            
            print(f"🔍 Formatted result: {formatted[:200]}...")
            return formatted
        
        # Also try a more flexible pattern for different formats
        flexible_items = re.findall(r'(\d+)\.\s*([^:]+):\s*([^,\n\r]+)', raw_response, re.MULTILINE)
        
        if flexible_items:
            print(f"🔍 Found {len(flexible_items)} flexible items")
            formatted = f"**Query:** {original_query}\n\n**Results:**\n\n"
            formatted += "| Rank | Product Name | Value |\n|------|-------------|-------|\n"
            
            for rank, product, value in flexible_items:
                # Clean up the values
                product_clean = product.strip()
                value_clean = value.strip()
                
                # Try to format as currency if it's a number
                try:
                    # Remove commas and convert to float
                    value_numeric = float(value_clean.replace(',', ''))
                    value_formatted = f"{value_numeric:,}"
                except:
                    value_formatted = value_clean
                
                formatted += f"| {rank} | {product_clean} | {value_formatted} |\n"
            
            formatted += f"\n**Total Items Found:** {len(flexible_items)}"
            formatted += "\n\n✅ *Query executed successfully.*"
            
            print(f"🔍 Formatted result: {formatted[:200]}...")
            return formatted
        
        print("🔍 No numbered list pattern found")
        return None

    def process_query(self, agent, query: str) -> str:
        """
        Enhanced query processing with improved response handling
        """
        # Intercept schema/column/field questions about automobile data or the specific table id
        lowered = query.lower()
        table_id = "table_489a3f4d-95bd-49f0-b9a0-155646951328_20250628104004"
        if (
            ("automobile data" in lowered or "automobile table" in lowered or table_id in lowered)
            and ("column" in lowered or "field" in lowered or "schema" in lowered)
        ):
            special = self.get_special_table_info("automobile_data")
            if special:
                return (
                    f"Columns in the automobile data table ({special['table_name']}):\n"
                    + ", ".join(special['schema'])
                )
        # Rewrite queries about 'automobile data' to use the table id
        if ("automobile data" in lowered or "automobile table" in lowered) and table_id not in lowered:
            # Replace 'automobile data' or 'automobile table' with the table id
            query = query.replace("automobile data", table_id)
            query = query.replace("automobile table", table_id)
        # Force table id and column names for any query about brands, regions, quarters, sales, or automobile
        import re
        keywords = ["bmw", "audi", "brand", "region", "quarter", "sales", "automobile"]
        if any(k in lowered for k in keywords):
            # Replace any table id pattern with the correct table id
            query = re.sub(r'table_[a-z0-9\-]+', table_id, query)
            # Replace common column name variants with your schema's column names
            column_map = {
                "Brand_Name": "Brand",
                "Date_of_Sale": "Date",
                "Units_Sold": "Units Sold",
                "Total_Sales_Value": "Total Revenue (USD)",
                "Total_Units": "Units Sold",
                "Total_Sales": "Total Revenue (USD)"
            }
            for wrong, right in column_map.items():
                query = query.replace(wrong, right)
        # Debug print the final query/SQL
        print(f"[DEBUG] Final query to execute: {query}")
        try:
            print(f"🔍 Processing query: {query}")
            
            # Get raw response from agent
            raw_response = None
            
            if hasattr(agent, 'run'):
                try:
                    # Try to get the full response with intermediate steps
                    full_response = agent.invoke({"input": query})
                    
                    if isinstance(full_response, dict):
                        print(f"🔍 Full response keys: {list(full_response.keys())}")
                        
                        # Check if we have intermediate steps with actual data
                        if "intermediate_steps" in full_response:
                            steps = full_response["intermediate_steps"]
                            print(f"🔍 Found {len(steps)} intermediate steps")
                            
                            # Look for the last tool output that contains actual data
                            for i, step in enumerate(reversed(steps)):
                                print(f"🔍 Step {len(steps)-i}: {step}")
                                if len(step) >= 2 and step[1]:
                                    tool_output = str(step[1])
                                    print(f"🔍 Tool output: {tool_output[:200]}...")
                                    
                                    # Check if this output contains actual data (numbers, products, etc.)
                                    if any(char.isdigit() for char in tool_output) or "(" in tool_output or "[" in tool_output:
                                        # Look for data patterns
                                        if re.search(r'\([^)]+\)', tool_output) or re.search(r'\[[^\]]+\]', tool_output):
                                            raw_response = tool_output
                                            print(f"🔍 Found actual data in step {len(steps)-i}: {raw_response[:200]}...")
                                            break
                                        # Also check for any data with colons and content
                                        elif ":" in tool_output and len(tool_output) > 50:
                                            raw_response = tool_output
                                            print(f"🔍 Found data with content in step {len(steps)-i}: {raw_response[:200]}...")
                                            break
                        
                        # If no intermediate data found, use the output
                        if not raw_response and "output" in full_response:
                            raw_response = full_response["output"]
                            print(f"🔍 Using output from full_response: {raw_response[:200]}...")
                    else:
                        # Fallback to regular run
                        raw_response = agent.run(query)
                        print(f"🔍 Using agent.run result: {raw_response[:200]}...")
                        
                except Exception as parsing_error:
                    # Handle parsing errors gracefully
                    error_str = str(parsing_error)
                    if "Could not parse LLM output:" in error_str:
                        # Extract the actual output from the parsing error
                        start_idx = error_str.find("Could not parse LLM output: `") + len("Could not parse LLM output: `")
                        end_idx = error_str.find("`\nFor troubleshooting")
                        if start_idx > 0 and end_idx > start_idx:
                            raw_output = error_str[start_idx:end_idx]
                            print(f"🔍 Extracted raw output from parsing error: {raw_output[:200]}...")
                            
                            # Check if the raw output contains numbered list data
                            if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', raw_output, re.MULTILINE):
                                print("🔍 Found numbered list in parsing error, returning raw data...")
                                # Remove "Final Answer:" and everything after it
                                if "Final Answer:" in raw_output:
                                    final_answer_start = raw_output.find("Final Answer:")
                                    raw_output = raw_output[:final_answer_start].strip()
                                return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully.*"
                            elif "The oil and gas data includes the following:" in raw_output:
                                # Extract everything after this line
                                data_start = raw_output.find("The oil and gas data includes the following:")
                                if data_start != -1:
                                    actual_data = raw_output[data_start:]
                                    print(f"🔍 Found oil and gas data: {actual_data[:200]}...")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in actual_data:
                                        final_answer_start = actual_data.find("Final Answer:")
                                        actual_data = actual_data[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{actual_data}\n\n✅ *Query executed successfully.*"
                            else:
                                # Try to extract any numbered list from the raw output
                                if re.search(r'\d+\.\s*[^:]+:\s*[^,\n]+', raw_output):
                                    print("🔍 Found numbered list pattern in raw output, returning raw data...")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in raw_output:
                                        final_answer_start = raw_output.find("Final Answer:")
                                        raw_output = raw_output[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully.*"
                                else:
                                    return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully (parsing error handled).*"
                        else:
                            raw_response = f"Query executed but with parsing issues: {error_str}"
                    else:
                        raw_response = f"Query error: {str(parsing_error)}"
            else:
                langchain_response = agent.invoke({"input": query})
                if isinstance(langchain_response, dict) and "output" in langchain_response:
                    raw_response = langchain_response["output"]
                else:
                    raw_response = str(langchain_response)
            
            print(f"🔍 Final raw response: {raw_response[:200]}...")
            
            # Check if the response contains actual data or is just a summary
            if raw_response and ("I don't know" in raw_response or "don't know" in raw_response.lower() or "has been listed above" in raw_response):
                # The LLM gave a summary instead of the data, try to extract from intermediate steps
                print("⚠️ LLM returned summary instead of data, checking intermediate steps...")
                
                try:
                    full_response = agent.invoke({"input": query})
                    if isinstance(full_response, dict) and "intermediate_steps" in full_response:
                        steps = full_response["intermediate_steps"]
                        for step in steps:
                            if len(step) >= 2 and step[1]:
                                tool_output = str(step[1])
                                # Look for actual data in tool outputs
                                if "Query results from" in tool_output and len(tool_output) > 100:
                                    # Extract the actual results part
                                    results_start = tool_output.find("Query results from")
                                    if results_start != -1:
                                        actual_results = tool_output[results_start:]
                                        print(f"🔍 Found actual results in intermediate step: {actual_results[:200]}...")
                                        return self._basic_format(actual_results, query)
                except Exception as e:
                    print(f"⚠️ Error extracting from intermediate steps: {e}")
            
            # Check if the response contains "Action Input:" followed by actual data
            if raw_response and "Action Input:" in raw_response:
                print("🔍 Found Action Input data, extracting...")
                action_input_start = raw_response.find("Action Input:")
                if action_input_start != -1:
                    action_data = raw_response[action_input_start:]
                    # Look for the actual query results after Action Input
                    if "[" in action_data and "(" in action_data:
                        # Extract the tuple data
                        start_bracket = action_data.find("[")
                        end_bracket = action_data.rfind("]")
                        if start_bracket != -1 and end_bracket != -1:
                            tuple_data = action_data[start_bracket:end_bracket+1]
                            print(f"🔍 Extracted tuple data: {tuple_data[:200]}...")
                            return self._basic_format(tuple_data, query)
            
            # Use basic formatting to show the actual data
            if raw_response:
                return self._basic_format(raw_response, query)
            else:
                return "No response generated"
                
        except Exception as e:
            print(f"❌ Query processing error: {e}")
            return f"Error processing query: {str(e)}"

    def _smart_response_processing(self, agent, query: str) -> str:
        """
        Smart response processing that handles various response formats including numbered lists
        """
        try:
            # Get the complete response from the agent
            complete_response = None
            
            if hasattr(agent, 'run'):
                try:
                    full_response = agent.invoke({"input": query})
                    
                    if isinstance(full_response, dict):
                        print(f"🔍 Full response keys: {list(full_response.keys())}")
                        
                        # Check if we have intermediate steps with actual data
                        if "intermediate_steps" in full_response:
                            steps = full_response["intermediate_steps"]
                            print(f"🔍 Found {len(steps)} intermediate steps")
                            
                            # Look for the complete response including numbered lists
                            for i, step in enumerate(steps):
                                print(f"🔍 Step {i}: {step}")
                                if len(step) >= 2 and step[1]:
                                    tool_output = str(step[1])
                                    print(f"🔍 Tool output: {tool_output[:200]}...")
                                    
                                    # Check if this contains the complete response with numbered list
                                    if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', tool_output, re.MULTILINE):
                                        complete_response = tool_output
                                        print(f"🔍 Found complete response with numbered list in step {i}")
                                        break
                                    # Also check for any response that contains both numbered list and final answer
                                    elif "Final Answer:" in tool_output and re.search(r'^\d+\.', tool_output, re.MULTILINE):
                                        complete_response = tool_output
                                        print(f"🔍 Found complete response with numbered list and final answer in step {i}")
                                        break
                        
                        # If no complete response found in intermediate steps, use the output
                        if not complete_response and "output" in full_response:
                            complete_response = full_response["output"]
                            print(f"🔍 Using output from full_response: {complete_response[:200]}...")
                            
                            # Check if the output only contains a summary but we have numbered data in intermediate steps
                            if ("have been listed above" in complete_response or "listed above" in complete_response or "have been provided above" in complete_response) and steps:
                                print("🔍 Output contains summary, checking intermediate steps for actual data...")
                                # Look for numbered list data in any intermediate step
                                for i, step in enumerate(steps):
                                    if len(step) >= 2 and step[1]:
                                        tool_output = str(step[1])
                                        print(f"🔍 Step {i} output: {tool_output[:200]}...")
                                        # Look for numbered list pattern
                                        if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', tool_output, re.MULTILINE):
                                            print(f"🔍 Found numbered list in step {i}, using this instead of summary")
                                            # Remove "Final Answer:" and everything after it
                                            if "Final Answer:" in tool_output:
                                                final_answer_start = tool_output.find("Final Answer:")
                                                tool_output = tool_output[:final_answer_start].strip()
                                            return f"**Query:** {query}\n\n**Results:**\n\n{tool_output}\n\n✅ *Query executed successfully.*"
                                        # Also check for numbered list without colons
                                        elif re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', tool_output, re.MULTILINE):
                                            print(f"🔍 Found numbered list (no colon) in step {i}, using this instead of summary")
                                            # Remove "Final Answer:" and everything after it
                                            if "Final Answer:" in tool_output:
                                                final_answer_start = tool_output.find("Final Answer:")
                                                tool_output = tool_output[:final_answer_start].strip()
                                            return f"**Query:** {query}\n\n**Results:**\n\n{tool_output}\n\n✅ *Query executed successfully.*"
                                        # Also check for any content that contains numbered items
                                        elif re.search(r'\d+\.\s*[^:]+:\s*[^,\n]+', tool_output):
                                            print(f"🔍 Found numbered list pattern in step {i}, using this instead of summary")
                                            # Remove "Final Answer:" and everything after it
                                            if "Final Answer:" in tool_output:
                                                final_answer_start = tool_output.find("Final Answer:")
                                                tool_output = tool_output[:final_answer_start].strip()
                                            return f"**Query:** {query}\n\n**Results:**\n\n{tool_output}\n\n✅ *Query executed successfully.*"
                                print("🔍 No numbered list found in intermediate steps")
                    else:
                        complete_response = agent.run(query)
                        print(f"🔍 Using agent.run result: {complete_response[:200]}...")
                        
                except Exception as parsing_error:
                    error_str = str(parsing_error)
                    
                    # Check if it's an iteration limit error
                    if "iteration limit" in error_str.lower() or "time limit" in error_str.lower():
                        print("⚠️ Agent hit iteration limit, trying direct SQL approach...")
                        # Try a more direct approach with a simpler query
                        try:
                            # Extract the database from the agent
                            if hasattr(agent, 'db'):
                                db = agent.db
                                # Try a simple query to get table information first
                                tables = db.get_usable_table_names()
                                if tables:
                                    # Look for oil/gas related tables
                                    oil_gas_tables = [t for t in tables if any(keyword in t.lower() for keyword in ['oil', 'gas', 'energy', 'petroleum'])]
                                    if oil_gas_tables:
                                        # Try to get data from the first oil/gas table
                                        table_name = oil_gas_tables[0]
                                        simple_query = f"SELECT TOP 10 * FROM {table_name}"
                                        try:
                                            result = db.run(simple_query)
                                            return f"**Query:** {query}\n\n**Results from {table_name}:**\n\n{result}\n\n✅ *Query executed successfully (retrieved sample data due to iteration limits).*"
                                        except Exception as e:
                                            return f"**Query:** {query}\n\n**Available oil/gas tables:** {', '.join(oil_gas_tables)}\n\n**Error:** {str(e)}\n\n⚠️ *Agent hit iteration limits. Please try a more specific query.*"
                                    else:
                                        return f"**Query:** {query}\n\n**Available tables:** {', '.join(tables)}\n\n⚠️ *No oil/gas specific tables found. Agent hit iteration limits. Please try a more specific query.*"
                                else:
                                    return f"**Query:** {query}\n\n**Error:** No tables available\n\n⚠️ *Agent hit iteration limits. Please check database connection.*"
                            else:
                                return f"**Query:** {query}\n\n**Error:** {error_str}\n\n⚠️ *Agent hit iteration limits. Please try a more specific query.*"
                        except Exception as retry_error:
                            return f"**Query:** {query}\n\n**Error:** {error_str}\n\n**Retry Error:** {str(retry_error)}\n\n⚠️ *Agent hit iteration limits and retry failed. Please try a more specific query.*"
                    
                    # Check if it's a parsing error with actual data
                    if "Could not parse LLM output:" in error_str:
                        print("🔍 Found parsing error, extracting actual data...")
                        start_idx = error_str.find("Could not parse LLM output: `") + len("Could not parse LLM output: `")
                        end_idx = error_str.find("`\nFor troubleshooting")
                        if start_idx > 0 and end_idx > start_idx:
                            raw_output = error_str[start_idx:end_idx]
                            print(f"🔍 Extracted raw output: {raw_output[:200]}...")
                            
                            # Check if the raw output contains numbered list data
                            if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', raw_output, re.MULTILINE):
                                print("🔍 Found numbered list in parsing error, returning raw data...")
                                # Remove "Final Answer:" and everything after it
                                if "Final Answer:" in raw_output:
                                    final_answer_start = raw_output.find("Final Answer:")
                                    raw_output = raw_output[:final_answer_start].strip()
                                return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully.*"
                            elif "The oil and gas data includes the following:" in raw_output:
                                # Extract everything after this line
                                data_start = raw_output.find("The oil and gas data includes the following:")
                                if data_start != -1:
                                    actual_data = raw_output[data_start:]
                                    print(f"🔍 Found oil and gas data: {actual_data[:200]}...")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in actual_data:
                                        final_answer_start = actual_data.find("Final Answer:")
                                        actual_data = actual_data[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{actual_data}\n\n✅ *Query executed successfully.*"
                            else:
                                # Try to extract any numbered list from the raw output
                                if re.search(r'\d+\.\s*[^:]+:\s*[^,\n]+', raw_output):
                                    print("🔍 Found numbered list pattern in raw output, returning raw data...")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in raw_output:
                                        final_answer_start = raw_output.find("Final Answer:")
                                        raw_output = raw_output[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully.*"
                                else:
                                    return f"**Query:** {query}\n\n**Results:**\n\n{raw_output}\n\n✅ *Query executed successfully (parsing error handled).*"
                        else:
                            raw_response = f"Query executed but with parsing issues: {error_str}"
                    else:
                        raw_response = f"Query error: {str(parsing_error)}"
            else:
                langchain_response = agent.invoke({"input": query})
                if isinstance(langchain_response, dict) and "output" in langchain_response:
                    complete_response = langchain_response["output"]
                else:
                    complete_response = str(langchain_response)
            
            print(f"🔍 Complete response: {complete_response[:500]}...")
            
            # Check if response indicates iteration limit
            if complete_response and ("iteration limit" in complete_response.lower() or "time limit" in complete_response.lower() or "Agent stopped due to" in complete_response):
                print("⚠️ Response indicates iteration limit, trying retry...")
                # Try a simpler approach
                try:
                    if hasattr(agent, 'db'):
                        db = agent.db
                        tables = db.get_usable_table_names()
                        if tables:
                            # Try to get basic table info
                            return f"**Query:** {query}\n\n**Available tables:** {', '.join(tables)}\n\n⚠️ *Agent hit iteration limits. Please try a more specific query like 'SELECT * FROM table_name LIMIT 10'*"
                except Exception as e:
                    pass
                
                return f"**Query:** {query}\n\n**Error:** Agent stopped due to iteration limits\n\n⚠️ *Please try a more specific query or break down your request into smaller parts.*"
            
            # Check if the response only contains a summary but we should have actual data
            if complete_response and ("have been listed above" in complete_response or "listed above" in complete_response or "have been provided above" in complete_response):
                print("🔍 Response contains summary, trying to extract actual data from intermediate steps...")
                try:
                    # Try to get the full response again to access intermediate steps
                    full_response = agent.invoke({"input": query})
                    if isinstance(full_response, dict) and "intermediate_steps" in full_response:
                        steps = full_response["intermediate_steps"]
                        print(f"🔍 Found {len(steps)} intermediate steps to search through")
                        for i, step in enumerate(steps):
                            if len(step) >= 2 and step[1]:
                                tool_output = str(step[1])
                                print(f"🔍 Step {i} output: {tool_output[:200]}...")
                                # Look for numbered list pattern
                                if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', tool_output, re.MULTILINE):
                                    print("🔍 Found numbered list in intermediate step, using this instead of summary")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in tool_output:
                                        final_answer_start = tool_output.find("Final Answer:")
                                        tool_output = tool_output[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{tool_output}\n\n✅ *Query executed successfully.*"
                                # Also check for any content that contains numbered items
                                elif re.search(r'\d+\.\s*[^:]+:\s*[^,\n]+', tool_output):
                                    print("🔍 Found numbered list pattern in step, using this instead of summary")
                                    # Remove "Final Answer:" and everything after it
                                    if "Final Answer:" in tool_output:
                                        final_answer_start = tool_output.find("Final Answer:")
                                        tool_output = tool_output[:final_answer_start].strip()
                                    return f"**Query:** {query}\n\n**Results:**\n\n{tool_output}\n\n✅ *Query executed successfully.*"
                        print("🔍 No numbered list found in intermediate steps")
                except Exception as e:
                    print(f"⚠️ Error extracting from intermediate steps: {e}")
            
            # First, try to extract numbered list data from the complete response
            numbered_list_result = self._extract_numbered_list_data(complete_response, query)
            if numbered_list_result:
                return numbered_list_result
            
            # If no numbered list found, check if the response contains "Final Answer:" and extract the data before it
            if complete_response and "Final Answer:" in complete_response:
                print("🔍 Found Final Answer, extracting data before it...")
                final_answer_start = complete_response.find("Final Answer:")
                if final_answer_start > 0:
                    data_before_final = complete_response[:final_answer_start].strip()
                    print(f"🔍 Data before Final Answer: {data_before_final[:200]}...")
                    
                    # Try to extract numbered list from the data before Final Answer
                    numbered_list_result = self._extract_numbered_list_data(data_before_final, query)
                    if numbered_list_result:
                        return numbered_list_result
            
            # If still no numbered list found, fall back to regular processing
            # --- NEW FALLBACK: Always return last tool output with numbered list if present ---
            try:
                full_response = agent.invoke({"input": query})
                if isinstance(full_response, dict) and "intermediate_steps" in full_response:
                    steps = full_response["intermediate_steps"]
                    for step in reversed(steps):
                        if len(step) >= 2 and step[1]:
                            tool_output = str(step[1])
                            # Look for numbered list pattern
                            if re.search(r'^\d+\.\s*[^:]+:\s*[^,\n]+', tool_output, re.MULTILINE):
                                # Return the full block, including "Final Answer" if present
                                return tool_output
            except Exception as e:
                print(f"Error extracting full output: {e}")
            return self.process_query(agent, query)
                
        except Exception as e:
            print(f"❌ Smart processing error: {e}")
            return f"Error processing query: {str(e)}"
    
    def get_database_info(self) -> dict:
        info = {}
        for db_name, db in self.databases.items():
            if isinstance(db, SQLDatabase):
                try:
                    tables = db.get_usable_table_names()
                    info[db_name] = {
                        'type': 'SQL Server',
                        'tables': tables,
                        'table_count': len(tables)
                    }
                except Exception as e:
                    info[db_name] = {'type': 'SQL Server', 'error': str(e)}
            elif isinstance(db, dict) and 'database' in db:
                try:
                    collections = db['database'].list_collection_names()
                    info[db_name] = {
                        'type': 'MongoDB',
                        'collections': collections,
                        'collection_count': len(collections)
                    }
                except Exception as e:
                    info[db_name] = {'type': 'MongoDB', 'error': str(e)}
        return info
    
    def _basic_format(self, raw_response: str, original_query: str) -> str:
        """
        Basic formatting for database responses
        """
        import re
        
        formatted = f"**Query:** {original_query}\n\n**Results:**\n\n"
        
        # Handle tuple/list data format like [('Product A', 123.45), ('Product B', 678.90)]
        if re.search(r'\[\([^)]+,\s*[^)]+\),?\s*\([^)]+,\s*[^)]+\),?', raw_response):
            # Extract tuples from the response
            tuples_match = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
            if tuples_match:
                tuples_str = tuples_match.group(1)
                # Parse the tuples with two values
                tuples = re.findall(r'\(([^,]+),\s*([^)]+)\)', tuples_str)
                formatted += "| # | Product Name | Value |\n|----|-------------|-------|\n"
                for i, (product, value) in enumerate(tuples, 1):
                    # Clean up the product name and value (remove quotes and extra spaces)
                    product_clean = product.strip().strip("'\"")
                    value_clean = value.strip().strip("'\"")
                    # Try to format as currency if it's a number
                    try:
                        float_val = float(value_clean)
                        value_formatted = f"{float_val:,}"
                    except:
                        value_formatted = value_clean
                    formatted += f"| {i} | {product_clean} | {value_formatted} |\n"
                formatted += f"\n**Total Items Found:** {len(tuples)}"
            else:
                formatted += raw_response
        # Handle tuple/list data format like [('Product A',), ('Product B',)]
        elif re.search(r'\[\([^)]+\),?\s*\([^)]+\),?', raw_response):
            # Extract tuples from the response
            tuples_match = re.search(r'\[(.*?)\]', raw_response, re.DOTALL)
            if tuples_match:
                tuples_str = tuples_match.group(1)
                # Parse the tuples
                tuples = re.findall(r'\(([^)]+)\)', tuples_str)
                formatted += "| # | Product Name |\n|----|-------------|\n"
                for i, product in enumerate(tuples, 1):
                    # Clean up the product name (remove quotes and extra spaces)
                    product_clean = product.strip().strip("'\"")
                    formatted += f"| {i} | {product_clean} |\n"
                formatted += f"\n**Total Products Found:** {len(tuples)}"
            else:
                formatted += raw_response
        # If it looks like a numbered list with data, format it nicely
        elif re.search(r'^\d+\.', raw_response, re.MULTILINE):
            lines = raw_response.split('\n')
            formatted += "| Rank | Product | Value |\n|------|---------|-------|\n"
            
            for line in lines:
                if re.match(r'^\d+\.', line.strip()):
                    # Extract rank, product name, and value
                    match = re.match(r'^(\d+)\.\s*([^:]+):\s*(.+)', line.strip())
                    if match:
                        rank, product, value = match.groups()
                        formatted += f"| {rank} | {product.strip()} | {value.strip()} |\n"
                elif line.strip() and not line.startswith('Final Answer:'):
                    formatted += f"{line}\n"
        else:
            # Just clean up the formatting
            formatted += raw_response.replace('Final Answer:', '\n**Summary:**')
        
        # Add some basic status
        if "Error" in raw_response or "error" in raw_response:
            formatted += "\n\n⚠️ *There was an issue with your query. Please check the error message above.*"
        elif raw_response.strip() == "":
            formatted += "\n\nℹ️ *No results returned for this query.*"
        else:
            formatted += "\n\n✅ *Query executed successfully.*"
            
        return formatted

    def get_special_table_info(self, key="automobile_data"):
        """Return the special table mapping for a given key (default: automobile_data)."""
        return self.special_tables.get(key, {})

# =====================
# MongoDB Agent Section (from du1.py)
# =====================

import websockets
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId
from bson.json_util import dumps, loads
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from langchain.callbacks.base import AsyncCallbackHandler
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class MongoDBConfig:
    """MongoDB and Azure OpenAI configuration"""
    def __init__(self):
        self.MONGODB_URI = "mongodb+srv://a2kafshan:admin%401234@cluster0.9ypdp80.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
        self.AZURE_OPENAI_API_KEY = "1NRS0AOwgsjCvmmes6eDNIRuw9TNtMzuzdtQE13P5IkVfeliDUSGJQQJ99BFAC77bzfXJ3w3AAABACOGJS8x"
        self.AZURE_OPENAI_ENDPOINT = "https://gen-ai-llm-deployment.openai.azure.com"
        self.AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
        self.AZURE_OPENAI_API_VERSION = "2025-01-01-preview"
        self._validate_config()
    def _validate_config(self):
        required_vars = [
            "MONGODB_URI", "AZURE_OPENAI_API_KEY", 
            "AZURE_OPENAI_ENDPOINT"
        ]
        missing_vars = [var for var in required_vars if not getattr(self, var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

class MongoDBManager:
    """MongoDB connection and operations manager"""
    def __init__(self, uri: str):
        self.uri = uri
        self.client = None
        self.db = None
    def connect(self, db_name: str = None):
        try:
            self.client = MongoClient(self.uri)
            self.client.admin.command('ping')
            if db_name:
                self.db = self.client[db_name]
            logger.info("Successfully connected to MongoDB")
        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    def get_database_names(self) -> List[str]:
        try:
            return self.client.list_database_names()
        except PyMongoError as e:
            logger.error(f"Error getting database names: {e}")
            return []
    def get_collection_names(self, db_name: str) -> List[str]:
        try:
            db = self.client[db_name]
            return db.list_collection_names()
        except PyMongoError as e:
            logger.error(f"Error getting collection names: {e}")
            return []
    def serialize_result(self, result: Any) -> str:
        if isinstance(result, (list, dict)):
            return dumps(result)
        return str(result)

class MongoDBCreateTool(BaseTool):
    name: str = "mongodb_create"
    description: str = """Insert one or multiple documents into MongoDB collection.\nUsage: mongodb_create(database_name, collection_name, documents)\n- documents: single dict or list of dicts to insert\n- Returns: inserted document IDs"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, database_name: str, collection_name: str, documents: Union[Dict, List[Dict]]) -> str:
        try:
            db = self.mongo_manager.client[database_name]
            collection = db[collection_name]
            if isinstance(documents, dict):
                result = collection.insert_one(documents)
                return f"Inserted document with ID: {result.inserted_id}"
            elif isinstance(documents, list):
                result = collection.insert_many(documents)
                return f"Inserted {len(result.inserted_ids)} documents with IDs: {result.inserted_ids}"
            else:
                return "Error: documents must be a dict or list of dicts"
        except Exception as e:
            return f"Error inserting documents: {str(e)}"

class MongoDBReadTool(BaseTool):
    name: str = "mongodb_read"
    description: str = """Query documents from MongoDB collection.\nUsage: mongodb_read(database_name, collection_name, query, limit, projection)\n- query: MongoDB query filter (dict) - use {} for all documents\n- limit: max number of documents to return (optional)\n- projection: fields to include/exclude (optional dict)"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, database_name: str, collection_name: str, query: Dict = None, limit: Optional[int] = None, projection: Optional[Dict] = None) -> str:
        try:
            db = self.mongo_manager.client[database_name]
            collection = db[collection_name]
            if query is None:
                query = {}
            cursor = collection.find(query, projection)
            if limit:
                cursor = cursor.limit(limit)
            results = list(cursor)
            return self.mongo_manager.serialize_result(results)
        except Exception as e:
            return f"Error querying documents: {str(e)}"

class MongoDBUpdateTool(BaseTool):
    name: str = "mongodb_update"
    description: str = """Update documents in MongoDB collection.\nUsage: mongodb_update(database_name, collection_name, query, update, update_many)\n- query: filter to match documents to update\n- update: update operations (use $set, $push, etc.)\n- update_many: True to update all matching docs, False for first match only"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, database_name: str, collection_name: str, query: Dict, update: Dict, update_many: bool = False) -> str:
        try:
            db = self.mongo_manager.client[database_name]
            collection = db[collection_name]
            if update_many:
                result = collection.update_many(query, update)
                return f"Updated {result.modified_count} documents (matched: {result.matched_count})"
            else:
                result = collection.update_one(query, update)
                return f"Updated {result.modified_count} document (matched: {result.matched_count})"
        except Exception as e:
            return f"Error updating documents: {str(e)}"

class MongoDBDeleteTool(BaseTool):
    name: str = "mongodb_delete"
    description: str = """Delete documents from MongoDB collection.\nUsage: mongodb_delete(database_name, collection_name, query, delete_many)\n- query: filter to match documents to delete\n- delete_many: True to delete all matching docs, False for first match only"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, database_name: str, collection_name: str, query: Dict, delete_many: bool = False) -> str:
        try:
            db = self.mongo_manager.client[database_name]
            collection = db[collection_name]
            if delete_many:
                result = collection.delete_many(query)
                return f"Deleted {result.deleted_count} documents"
            else:
                result = collection.delete_one(query)
                return f"Deleted {result.deleted_count} document"
        except Exception as e:
            return f"Error deleting documents: {str(e)}"

class MongoDBSchemaInspectorTool(BaseTool):
    name: str = "mongodb_inspect"
    description: str = """Inspect MongoDB databases and collections structure.\nUsage: mongodb_inspect(action, database_name, collection_name)\n- action: 'databases', 'collections', 'sample', 'count'\n- For 'sample': returns sample documents from collection\n- For 'count': returns document count in collection"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, action: str, database_name: str = None, collection_name: str = None) -> str:
        try:
            if action == "databases":
                dbs = self.mongo_manager.get_database_names()
                return f"Available databases: {dbs}"
            elif action == "collections" and database_name:
                collections = self.mongo_manager.get_collection_names(database_name)
                return f"Collections in {database_name}: {collections}"
            elif action == "sample" and database_name and collection_name:
                db = self.mongo_manager.client[database_name]
                collection = db[collection_name]
                sample = list(collection.find().limit(3))
                return f"Sample documents from {collection_name}:\n{self.mongo_manager.serialize_result(sample)}"
            elif action == "count" and database_name and collection_name:
                db = self.mongo_manager.client[database_name]
                collection = db[collection_name]
                count = collection.count_documents({})
                return f"Document count in {collection_name}: {count}"
            else:
                return "Invalid action or missing parameters. Use: databases, collections, sample, or count"
        except Exception as e:
            return f"Error inspecting database: {str(e)}"

class MongoDBAggregateTool(BaseTool):
    name: str = "mongodb_aggregate"
    description: str = """Run aggregation pipeline on MongoDB collection.\nUsage: mongodb_aggregate(database_name, collection_name, pipeline)\n- pipeline: list of aggregation stages (e.g., [{"$match": {...}}, {"$group": {...}}])"""
    mongo_manager: MongoDBManager = Field(default=None, exclude=True)
    def __init__(self, mongo_manager: MongoDBManager, **kwargs):
        super().__init__(mongo_manager=mongo_manager, **kwargs)
    def _run(self, database_name: str, collection_name: str, pipeline: List[Dict]) -> str:
        try:
            db = self.mongo_manager.client[database_name]
            collection = db[collection_name]
            results = list(collection.aggregate(pipeline))
            return self.mongo_manager.serialize_result(results)
        except Exception as e:
            return f"Error running aggregation: {str(e)}"

class WebSocketCallbackHandler(AsyncCallbackHandler):
    def __init__(self, websocket):
        self.websocket = websocket
    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        try:
            await self.websocket.send(json.dumps({
                "type": "token",
                "content": token,
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error(f"Error sending token via WebSocket: {e}")

class MongoDBLangChainAgent:
    def __init__(self, config: MongoDBConfig):
        self.config = config
        self.mongo_manager = MongoDBManager(config.MONGODB_URI)
        self.llm = None
        self.agent_executor = None
        self._initialize_components()
    def _initialize_components(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=self.config.AZURE_OPENAI_ENDPOINT,
            api_key=self.config.AZURE_OPENAI_API_KEY,
            api_version=self.config.AZURE_OPENAI_API_VERSION,
            deployment_name=self.config.AZURE_OPENAI_DEPLOYMENT,
            temperature=0.1,
            streaming=True
        )
        self.mongo_manager.connect()
        self.tools = [
            MongoDBCreateTool(self.mongo_manager),
            MongoDBReadTool(self.mongo_manager),
            MongoDBUpdateTool(self.mongo_manager),
            MongoDBDeleteTool(self.mongo_manager),
            MongoDBSchemaInspectorTool(self.mongo_manager),
            MongoDBAggregateTool(self.mongo_manager)
        ]
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MongoDB database assistant with full CRUD capabilities. \nYou can help users interact with their MongoDB database using natural language.\n\nAvailable tools:\n- mongodb_create: Insert documents\n- mongodb_read: Query/find documents  \n- mongodb_update: Update documents\n- mongodb_delete: Delete documents\n- mongodb_inspect: Inspect database structure\n- mongodb_aggregate: Run aggregation pipelines\n\nGuidelines:\n1. Always inspect the database structure first if user asks about data\n2. For queries, ask for clarification if database/collection names aren't specified\n3. Use proper MongoDB query syntax in tool calls\n4. Handle ObjectId conversion when needed\n5. Provide clear explanations of operations performed\n6. Be helpful with query optimization suggestions\n\nWhen working with data:\n- Use {{}} for empty query (find all)\n- ObjectIds should be referenced as strings in queries\n- Use $set, $push, $pull operators appropriately for updates\n- Always confirm destructive operations\n"""),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}")
        ])
        agent = create_openai_tools_agent(self.llm, self.tools, prompt)
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=True,
            return_intermediate_steps=True,
            max_iterations=5
        )
    async def process_query(self, query: str, websocket=None) -> Dict[str, Any]:
        try:
            callbacks = []
            if websocket:
                callbacks.append(WebSocketCallbackHandler(websocket))
            response = await self.agent_executor.ainvoke(
                {"input": query},
                config={"callbacks": callbacks} if callbacks else None
            )
            serialized_steps = []
            if "intermediate_steps" in response:
                for step in response["intermediate_steps"]:
                    if hasattr(step, '_iter_') and len(step) == 2:
                        action, observation = step
                        serialized_steps.append({
                            "tool": getattr(action, 'tool', str(action)),
                            "tool_input": getattr(action, 'tool_input', {}),
                            "log": getattr(action, 'log', ''),
                            "observation": str(observation)
                        })
                    else:
                        serialized_steps.append(str(step))
            return {
                "success": True,
                "response": response["output"],
                "intermediate_steps": serialized_steps,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# =====================
# End MongoDB Agent Section
# =====================

# FastAPI App
app = FastAPI()

# Global agent
db_agent = None

@app.on_event("startup")
async def startup():
    global db_agent
    global primary_db_agent
    print("🚀 Starting Enhanced Multi-Database Agent...")
    azure_config = AzureOpenAIConfig(
        api_key="1NRS0AOwgsjCvmmes6eDNIRuw9TNtMzuzdtQE13P5IkVfeliDUSGJQQJ99BFAC77bzfXJ3w3AAABACOGJS8x",
        endpoint="https://gen-ai-llm-deployment.openai.azure.com",
        deployment="gpt-4o",
        api_version="2025-01-01-preview"
    )
    db_agent = DatabaseAgent(azure_config)
    
    # Add SQL Server database
    config = DatabaseConfig(
        host="research-db-pulsee.database.windows.net",
        port="1433",
        database="secondary_research_db",
        username="Sql_server",
        password="My-Admin"
    )
    success = db_agent.add_sql_database("secondary_db", config)
    
    # Debug: Print environment variables
    main_db_host = "genflow-postgress.postgres.database.azure.com"
    main_db_name = "decision-pulse-dev-dataset"
    main_db_user = "azureuser"
    main_db_password = "530228@mka"
    
    print(f"🔍 Database connection values:")
    print(f"   MAIN_DB_HOST: {main_db_host}")
    print(f"   MAIN_DB_NAME: {main_db_name}")
    print(f"   MAIN_DB_USER: {main_db_user}")
    print(f"   MAIN_DB_PASSWORD: {'***' if main_db_password else 'None'}")
    
    # Primary database connection using PostgreSQL
    primary_config = DatabaseConfig(
        host=main_db_host,
        port="5432",
        database=main_db_name,
        username=main_db_user,
        password=main_db_password
    )
    
    try:
        db_agent.add_postgres_database("primary_db", primary_config)
        print(f"✅ Connected to primary_db")
        # Create and assign the primary_db_agent here
        llm = db_agent._create_llm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        sql_db = db_agent.databases["primary_db"]
        from langchain.agents import create_sql_agent, AgentType
        global primary_db_agent
        primary_db_agent = create_sql_agent(
            llm=llm,
            db=sql_db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=50,
            return_intermediate_steps=True
        )
    except Exception as e:
        print(f"❌ Failed to connect to primary_db: {e}")
    
    # Uncomment to add MongoDB
    mongodb_connection = "mongodb+srv://a2kafshan:admin%401234@cluster0.9ypdp80.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    db_agent.add_mongodb_database("mongo_db", mongodb_connection, "your_database_name")
    
    if success:
        print(f"✅ Enhanced Agent ready with {len(db_agent.tools)} tools!")
    else:
        print("❌ No databases connected!")

@app.get("/")
async def root():
    tool_count = len(db_agent.tools) if db_agent else 0
    db_count = len(db_agent.databases) if db_agent else 0
    return {
        "message": "Enhanced WebSocket Multi-Database Agent", 
        "websocket_endpoint": "/ws",
        "databases_connected": db_count,
        "tools_available": tool_count,
        "status": "ready" if tool_count > 0 else "no_databases",
        "features": ["LLM Response Formatting", "Enhanced Error Handling", "Multi-Database Support"]
    }

@app.get("/info")
async def get_info():
    if db_agent:
        return db_agent.get_database_info()
    return {"error": "Agent not initialized"}

# Helper function to aggregate and format results from all DBs
async def run_pipeline_query(query, filters=None, multi_db_agent=None, db_agent=None, mongodb_agent=None, primary_db_agent=None):
    results = {}
    # SQL/PG (multi_db)
    sql_query = query
    if filters:
        filter_clauses = []
        for k, v in filters.items():
            filter_clauses.append(f"{k} = '{v}'")
        if filter_clauses:
            if "where" in sql_query.lower():
                sql_query += " AND " + " AND ".join(filter_clauses)
            else:
                sql_query += " WHERE " + " AND ".join(filter_clauses)
    if multi_db_agent and db_agent:
        try:
            sql_result = db_agent._smart_response_processing(multi_db_agent, sql_query)
            results["SQL/PG"] = sql_result
        except Exception as e:
            results["SQL/PG"] = f"Error: {str(e)}"
    # Primary DB (PostgreSQL only)
    if primary_db_agent:
        try:
            primary_result = db_agent._smart_response_processing(primary_db_agent, sql_query)
            results["Primary DB"] = primary_result
        except Exception as e:
            results["Primary DB"] = f"Error: {str(e)}"
    # MongoDB
    mongo_query = query
    if filters:
        filter_str = ", ".join([f"{k}: {v}" for k, v in filters.items()])
        mongo_query = f"{mongo_query} with filters ({filter_str})"
    if mongodb_agent:
        try:
            mongo_result = await mongodb_agent.process_query(mongo_query)
            results["MongoDB"] = mongo_result.get("response", str(mongo_result))
        except Exception as e:
            results["MongoDB"] = f"Error: {str(e)}"
    # Format combined result
    combined = """# Pipeline Results (All Databases)\n"""
    for db, res in results.items():
        combined += f"\n---\n**{db} Result:**\n\n{res}\n"
    return combined

def get_primary_db_agent(db_agent):
    if db_agent and "primary_db" in db_agent.databases:
        llm = db_agent._create_llm()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        sql_db = db_agent.databases["primary_db"]
        return create_sql_agent(
            llm=llm,
            db=sql_db,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=50,
            return_intermediate_steps=True
        )
    return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global primary_db_agent
    await websocket.accept()
    print("🔌 WebSocket connected!")
    try:
        await websocket.send_text(json.dumps({
            "type": "connected", 
            "data": {
                "message": "Connected to Enhanced Multi-Database Agent!",
                "databases": len(db_agent.databases) if db_agent else 0,
                "tools": len(db_agent.tools) if db_agent else 0,
                "features": ["Smart Response Formatting", "Enhanced Error Handling"],
                "modes": ["multi_db", "primary_db", "pipeline"]
            }
        }))
        
        if not db_agent or not db_agent.tools:
            await websocket.send_text(json.dumps({
                "type": "error",
                "data": "No database connections available. Check server startup logs."
            }))
            return
            
        # Create agents for both modes
        multi_db_agent = db_agent.create_agent()
        
        # Create primary_db only agent
        primary_db_agent = None
        if "primary_db" in db_agent.databases:
            llm = db_agent._create_llm()
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            sql_db = db_agent.databases["primary_db"]
            from langchain.agents import create_sql_agent, AgentType
            primary_db_agent = create_sql_agent(
                llm=llm,
                db=sql_db,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                memory=memory,
                handle_parsing_errors=True,
                max_iterations=50,
                return_intermediate_steps=True
            )
        
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": f"Invalid JSON: {str(e)}"
                }))
                continue
            if message.get("type") == "query":
                query = message.get("data", "")
                mode = message.get("mode", None)
                industry = message.get("industry", None)
                filters = {k: v for k, v in message.items() if k not in ["type", "data", "mode"]}
                print(f"📝 Processing query: {query} (mode: {mode}, filters: {filters})")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "status", 
                        "data": f"🔍 Processing your query..."
                    }))
                    agent_used = None
                    def is_sql_fallback(resp):
                        if not isinstance(resp, str):
                            return False
                        resp_l = resp.strip().lower()
                        return (
                            "don't know" in resp_l or
                            "not related" in resp_l or
                            "no data" in resp_l or
                            "no result" in resp_l or
                            "no matching" in resp_l or
                            resp_l in ["", "none", "null"] or
                            len(resp_l) < 10  # very short/empty
                        )
                    def inject_industry_sql(q, industry):
                        q_lower = q.lower()
                        if "where" in q_lower:
                            return q + f" AND industry = '{industry}'"
                        else:
                            return q + f" WHERE industry = '{industry}'"
                    def inject_industry_mongo(q, industry):
                        return q + f" for industry {industry}"
                    # --- PRIMARY DB ENFORCEMENT LOGIC ---
                    use_primary = False
                    if mode == "primary_db":
                        use_primary = True
                    elif "hr data" in query.lower():
                        use_primary = True
                    # --- END PRIMARY DB ENFORCEMENT LOGIC ---
                    if mode == "pipeline":
                        # Run pipeline query across all DBs with dynamic filters, including primary_db
                        combined = await run_pipeline_query(
                            query, filters=filters, 
                            multi_db_agent=multi_db_agent, db_agent=db_agent, mongodb_agent=mongodb_agent, primary_db_agent=primary_db_agent
                        )
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "data": combined
                        }))
                        continue
                    elif mode == "mongodb":
                        agent_used = "mongodb"
                        if industry:
                            query = inject_industry_mongo(query, industry)
                    elif use_primary:
                        agent_used = "primary_db"
                        if industry:
                            query = inject_industry_sql(query, industry)
                    elif mode == "multi_db":
                        agent_used = "multi_db"
                        if industry:
                            query = inject_industry_sql(query, industry)
                    elif not mode:
                        # Try multi_db agent first, but if query is about hr data, use primary_db_agent
                        if "hr data" in query.lower():
                            agent_used = "primary_db"
                        else:
                            agent_used = "multi_db"
                    # Process query for explicit mode
                    if agent_used == "mongodb":
                        result = await mongodb_agent.process_query(query)
                        await websocket.send_text(json.dumps({
                            "type": "response",
                            "data": result.get("response", "No response")
                        }))
                        continue
                    elif agent_used == "primary_db":
                        if primary_db_agent:
                            response = db_agent._smart_response_processing(primary_db_agent, query)
                        else:
                            response = "**Error:** Primary database agent not available. Check server startup logs."
                        await websocket.send_text(json.dumps({
                            "type": "response", 
                            "data": response
                        }))
                        continue
                    elif agent_used == "multi_db":
                        response = db_agent._smart_response_processing(multi_db_agent, query)
                        await websocket.send_text(json.dumps({
                            "type": "response", 
                            "data": response
                        }))
                        continue
                    else:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "data": "Could not determine which database to use. Please specify 'mode'."
                        }))
                        continue
                except Exception as e:
                    print(f"❌ Query error: {e}")
                    await websocket.send_text(json.dumps({
                        "type": "error", 
                        "data": f"Query error: {str(e)}"
                    }))
                    continue
            elif message.get("type") == "database_info":
                try:
                    info = db_agent.get_database_info()
                    mode = message.get("mode", "multi_db")
                    if mode == "primary_db":
                        # Return only primary_db info
                        filtered_info = {"primary_db": info.get("primary_db")}
                        await websocket.send_text(json.dumps({
                            "type": "database_info",
                            "data": filtered_info
                        }))
                    else:
                        # Return all database info
                        await websocket.send_text(json.dumps({
                            "type": "database_info",
                            "data": info
                        }))
                except Exception as e:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": f"Info error: {str(e)}"
                    }))
                    
            elif message.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong", 
                    "data": "Enhanced Agent is alive and ready!"
                }))
                
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

# Initialize MongoDB agent and config once
mongodb_config = MongoDBConfig()
mongodb_agent = MongoDBLangChainAgent(mongodb_config)

@app.post("/query")
async def query_endpoint(request: Request):
    global primary_db_agent
    data = await request.json()
    query = data.get("query", "")
    mode = data.get("mode", None)
    industry = data.get("industry", None)
    if not query:
        return JSONResponse({"success": False, "error": "No query provided"}, status_code=400)
    if mode == "pipeline":
        # Create agents if not already created
        multi_db_agent = db_agent.create_agent() if db_agent else None
        combined = await run_pipeline_query(
            query, filters=None, 
            multi_db_agent=multi_db_agent, db_agent=db_agent, mongodb_agent=mongodb_agent, primary_db_agent=primary_db_agent
        )
        return {"success": True, "response": combined}
    # --- NEW LOGIC FOR SQL/PG ---
    if mode == "multi_db":
        agent = db_agent.create_agent()
        response = db_agent._smart_response_processing(agent, query)
        return {"success": True, "response": response}
    elif mode == "primary_db":
        if primary_db_agent:
            response = db_agent._smart_response_processing(primary_db_agent, query)
        else:
            response = "**Error:** Primary database agent not available. Check server startup logs."
        return {"success": True, "response": response}
    # --- END NEW LOGIC ---
    # Default: MongoDB
    result = await mongodb_agent.process_query(query)
    return JSONResponse(result)

if __name__ == "__main__":
    print("🚀 Starting Enhanced WebSocket Multi-Database Agent...")
    print("📡 WebSocket: ws://localhost:8000/ws")
    print("🌐 HTTP: http://localhost:8000")
    print("📊 Info: http://localhost:8000/info")
    print("✨ Features: Smart Response Formatting, Enhanced Error Handling")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
