from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3
import bcrypt
import traceback
import os
import joblib
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import re
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()




app = Flask(__name__)
CORS(app, supports_credentials=True)

# Get RapidAPI configuration from environment variables
API_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
HEADERS = {
    "x-rapidapi-key": os.environ.get("RAPIDAPI_KEY", ""),
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

# Configuration
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'cluster_cart.db')
AMAZON_DB_PATH = os.path.join(BASE_DIR, 'amazon_products.db')

# Load model artifacts
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
kmeans_model = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
training_columns = joblib.load(os.path.join(BASE_DIR, 'training_columns.pkl'))
numeric_cols = joblib.load(os.path.join(BASE_DIR, 'numeric_cols.pkl'))

# Ensure that training_columns does not include the target "Cluster"
if "Cluster" in training_columns:
    training_columns.remove("Cluster")

# Initialize Main Database
def init_db():
    """Initialize the main database tables if they don't exist"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        age INTEGER,
        gender TEXT,
        location TEXT,
        shopping_frequency TEXT,
        annual_income REAL,
        password TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create user_cluster table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_cluster (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        cluster INTEGER NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    logging.info("Main database initialized successfully")

# Initialize Amazon Products Database
def init_amazon_db():
    conn = sqlite3.connect(AMAZON_DB_PATH)
    cursor = conn.cursor()
    
    # Create an enhanced products table with all fields 
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asin TEXT UNIQUE,
        title TEXT,
        price REAL,
        original_price REAL,
        currency TEXT,
        rating REAL,
        reviews_count INTEGER,
        image_url TEXT,
        product_url TEXT,
        is_prime BOOLEAN,
        is_best_seller BOOLEAN,
        is_amazon_choice BOOLEAN,
        climate_pledge_friendly BOOLEAN,
        num_offers INTEGER,
        minimum_offer_price TEXT,
        sales_volume TEXT,
        delivery TEXT,
        has_variations BOOLEAN,
        search_query TEXT,
        created_at TIMESTAMP
    )
    ''')

        # Create wishlist table to store all product info along with email and cluster
    cursor.execute('''
CREATE TABLE IF NOT EXISTS wishlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL,
    asin TEXT NOT NULL,
    cluster INTEGER NOT NULL,
    title TEXT NOT NULL,
    price REAL NOT NULL,
    rating REAL,
    reviews_count INTEGER,
    image_url TEXT,
    product_url TEXT,
    is_prime BOOLEAN,
    is_best_seller BOOLEAN,
    is_amazon_choice BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(email, asin)  -- Add unique constraint
)
''')

    conn.commit()
    conn.close()
    logging.info("Enhanced database initialized successfully")

# Initialize database when app starts
init_db()  # Initialize main database
init_amazon_db() # Initialize Amazon products database

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_amazon_db_connection():
    conn = sqlite3.connect(AMAZON_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_user(email):
    """Fetch user data with proper column names matching training data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT 
                email,
                password,
                age AS Age,
                gender AS Gender,
                annual_income AS Income,
                location AS Location,
                shopping_frequency AS "Frequency of Purchases"
            FROM users 
            WHERE email = ?
        ''', (email,))
        user_row = cursor.fetchone()
        return user_row
    except Exception as e:
        return None
    finally:
        conn.close()

def preprocess_user_data(user_row):
    """Transform user data to match model training format"""
    try:
        user_data = dict(user_row)

        for key in ["email", "password"]:
            if key in user_data:
                del user_data[key]

        df = pd.DataFrame([user_data])
        
        df_encoded = pd.get_dummies(df, 
                                   columns=['Location', 'Frequency of Purchases'],
                                   drop_first=True)
        
        missing_cols = set(training_columns) - set(df_encoded.columns)
        
        for col in missing_cols:
            df_encoded[col] = 0
        df_encoded = df_encoded[training_columns]
        
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
        
        return df_encoded
    except Exception as e:
        return None

def predict_cluster(user_data):
    """Predict cluster for preprocessed data"""
    try:
        processed_data = preprocess_user_data(user_data)
        if processed_data is not None:
            cluster = kmeans_model.predict(processed_data)[0]
            return cluster
    except Exception as e:
        pass
    return None

def store_user_cluster(email, cluster):
    """Store user cluster in database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cluster_int = int(cluster)
        
        cursor.execute('''
            INSERT INTO user_cluster (email, cluster)
            VALUES (?, ?)
            ON CONFLICT(email) DO UPDATE SET cluster = excluded.cluster
        ''', (email, cluster_int))
        conn.commit()
    except Exception as e:
        pass
    finally:
        conn.close()

def get_hardcoded_recommendations(gender, cluster):
    """Return hardcoded product recommendations based on gender and cluster"""
    
    # Male recommendations
    male_recommendations = {
        0: ["jewelry for male", "pants for male", "boots for male", "sunglasses for male", "shorts for male"],
        1: ["watches for male", "shoes for male", "jackets for male", "backpacks for male", "hats for male"],  # Backup products for cluster 1
        2: ["shirt for male", "sweater for male", "belt for male", "pants for male", "gloves for male"],
        3: ["sneakers for male", "jeans for male", "hoodies for male", "wallets for male", "sunglasses for male"]  # Backup products for cluster 3
    }
    
    # Female recommendations
    female_recommendations = {
        0: ["dresses for female", "earrings for female", "scarves for female", "bracelets for female", "leggings for female"],  # Backup products for cluster 0
        1: ["sunglasses for female", "blouse for female", "boots for female", "socks for female", "shirt for female"],
        2: ["skirts for female", "dresses for female", "necklaces for female", "perfume for female", "makeup for female"],  # Backup products for cluster 2
        3: ["sandals for female", "handbag for female", "blouse for female", "belt for female", "shirt for female"]
    }
    
    # Convert gender to string if it's a number
    if isinstance(gender, int):
        gender = "female" if gender == 0 else "male"
    elif isinstance(gender, str) and gender.isdigit():
        gender = "female" if gender == "0" else "male"
    
    # Normalize gender string
    gender = gender.lower()
    
    # Return appropriate recommendations
    if gender == "male":
        return male_recommendations.get(cluster, male_recommendations[0])  # Default to cluster 0 if not found
    else:  # female
        return female_recommendations.get(cluster, female_recommendations[1])  # Default to cluster 1 if not found

# Add this new endpoint to your app.py file
@app.route('/add_to_wishlist', methods=['POST'])
def add_to_wishlist():
    try:
        data = request.json
        email = data.get('email')
        cluster = data.get('cluster')
        product = data.get('product')
        
        if not email or not cluster or not product:
            return jsonify({"error": "Email, cluster, and product details are required"}), 400

        # Check for required ASIN
        asin = product.get('asin')
        if not asin:
            return jsonify({"error": "Product ASIN is required"}), 400

        conn = get_amazon_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO wishlist (
                    email, cluster, asin, title, price, rating, 
                    reviews_count, image_url, product_url, 
                    is_prime, is_best_seller, is_amazon_choice
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                email, cluster, asin,
                product.get('title'), product.get('price'), product.get('rating'),
                product.get('reviews_count'), product.get('image_url'),
                product.get('product_url'), product.get('is_prime'),
                product.get('is_best_seller'), product.get('is_amazon_choice')
            ))
            conn.commit()
            return jsonify({"message": "Product added to wishlist"}), 201
        except sqlite3.IntegrityError:
            return jsonify({"error": "Product already in wishlist"}), 409
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Error adding to wishlist: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/wishlist.html')
def serve_wishlist_page():
    """Serve wishlist.html from the root directory"""
    return send_from_directory(BASE_DIR, 'wishlist.html')

@app.route('/get_user_cluster', methods=['GET'])
def get_user_cluster():
    """Endpoint to fetch the user's cluster number"""
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get user cluster from user_cluster table
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT cluster FROM user_cluster WHERE email = ?", (email,))
        cluster_row = cursor.fetchone()
        conn.close()

        if not cluster_row:
            return jsonify({"error": "User cluster not found"}), 404

        return jsonify({"cluster": cluster_row['cluster']}), 200

    except Exception as e:
        logging.error(f"Error fetching user cluster: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500



@app.route('/remove_from_wishlist', methods=['POST'])
def remove_from_wishlist():
    """Remove a product from the user's wishlist"""
    try:
        data = request.json
        email = data.get('email')
        product_id = data.get('product_id')
        
        if not email or not product_id:
            return jsonify({"error": "Email and product_id are required"}), 400

        # Delete from wishlist table
        conn = get_amazon_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM wishlist WHERE email = ? AND id = ?", (email, product_id))
        conn.commit()
        conn.close()

        return jsonify({"message": "Product removed from wishlist"}), 200

    except Exception as e:
        logging.error(f"Error removing from wishlist: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/get_wishlist', methods=['GET'])
def get_wishlist():
    """Get the user's wishlist"""
    try:
        email = request.args.get('email')
        
        if not email:
            return jsonify({"error": "Email is required"}), 400

        # Get wishlist items for the user
        conn = get_amazon_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM wishlist WHERE email = ?", (email,))
        wishlist_items = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return jsonify({"products": wishlist_items}), 200

    except Exception as e:
        logging.error(f"Error retrieving wishlist: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/get_category_products', methods=['GET'])
def get_category_products():
    """Endpoint for fetching all products for a specific category"""
    try:
        # Get parameters from query
        user_email = request.args.get('email')
        category = request.args.get('category')
        
        logging.info(f"GET /get_category_products request received for email: {user_email}, category: {category}")
        
        if not user_email or not category:
            logging.error("Email or category parameter is missing")
            return jsonify({"error": "Email and category parameters are required"}), 400

        # Get all products for this category (with higher limit)
        stored_products = get_stored_products(category, limit=50)  # Increased limit for "View All"
        
        if stored_products:
            logging.info(f"Found {len(stored_products)} products in database for {category}")
            return jsonify({"products": stored_products})
        else:
            logging.warning(f"No stored products found for {category}")
            return jsonify({"products": []})
            
    except Exception as e:
        logging.error(f"Server error in /get_category_products: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

def get_stored_products(search_query=None, limit=200):
    """Function for product retrieval from existing database"""
    conn = None
    
    logging.info(f"Retrieving stored products for query: {search_query}")
    
    try:
        conn = get_amazon_db_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if search_query:
         cursor.execute('''
        SELECT * FROM products 
        WHERE title LIKE ? OR search_query LIKE ?
        ORDER BY created_at DESC
        LIMIT ?
        ''', (f'%{search_query}%', f'%{search_query}%', limit))
        else:
            cursor.execute('''
            SELECT * FROM products 
            ORDER BY created_at DESC
            LIMIT ?
            ''', (limit,))
                
        products = [dict(row) for row in cursor.fetchall()]
        logging.info(f"Retrieved {len(products)} products from database")
        
        return products
    except Exception as e:
        logging.error(f"Error in get_stored_products: {str(e)}")
        return []
    finally:
        if conn:
            conn.close()

@app.route('/')
def serve_frontpage():
    """Serve frontpage.html as the default page"""
    return send_from_directory(BASE_DIR, 'frontpage.html')

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt())
        
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (full_name, email, age, gender, location, shopping_frequency, annual_income, password)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (data['full_name'], data['email'], data['age'], data['gender'], 
              data['location'], data['shopping_frequency'], data['annual_income'], 
              password.decode('utf-8')))
        conn.commit()
        conn.close()
        return jsonify({"message": "Signup Successful!"}), 201
    except sqlite3.IntegrityError:
        return jsonify({"error": "Email already registered!"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/signin_page.html')
def serve_signin_page():
    return send_from_directory(BASE_DIR, 'signin_page.html')

@app.route('/signin', methods=['POST'])
def signin():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be in JSON format"}), 400
            
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT email, password FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
            # Process user data and assign cluster if not already assigned
            user_data = get_user(email)
            if user_data:
                # Check if user already has a cluster
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT cluster FROM user_cluster WHERE email = ?", (email,))
                cluster_row = cursor.fetchone()
                conn.close()
                
                if not cluster_row:
                    # If no cluster assigned yet, predict and store it
                    cluster = predict_cluster(user_data)
                    if cluster is not None:
                        store_user_cluster(email, cluster)
            
            return jsonify({
                "message": "Signin successful!",
                "redirect": "product_page.html",
                "email": email  # Return email to client to use for product retrieval
            })
        else:
            return jsonify({"error": "Invalid email or password"}), 401

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/product_page.html')
def serve_product_page():
    return send_from_directory(BASE_DIR, 'product_page.html')

@app.route('/get_products', methods=['GET'])
def get_products():
    """Endpoint for fetching product recommendations from database only"""
    try:
        # Get email from query parameter
        user_email = request.args.get('email')
        logging.info(f"GET /get_products request received for email: {user_email}")
        
        if not user_email:
            logging.error("Email parameter is missing")
            return jsonify({"error": "Email parameter is required"}), 400

        # Get user gender from users table
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT gender FROM users WHERE email = ?", (user_email,))
            user = cursor.fetchone()
            
            if not user:
                logging.error(f"User not found for email: {user_email}")
                return jsonify({"error": "User not found"}), 404
            
            # Simplify gender parsing - more robust handling
            user_gender = user['gender']
            if isinstance(user_gender, str) and user_gender.isdigit():
                user_gender = int(user_gender)
                
            # Clear and consistent gender conversion
            if user_gender == 0 or user_gender == 'female':
                gender_str = "female"
            else:
                gender_str = "male"
                
            logging.info(f"User gender: {gender_str}")
            
        except Exception as e:
            logging.error(f"Error retrieving user gender: {str(e)}")
            return jsonify({"error": f"Error retrieving user data: {str(e)}"}), 500
        finally:
            if conn:
                conn.close()
                conn = None
        
        # Get user cluster from user_cluster table
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT cluster FROM user_cluster WHERE email = ?", (user_email,))
            cluster_row = cursor.fetchone()
            
            if not cluster_row:
                logging.info(f"No cluster found for user {user_email}, predicting now")
                # If cluster not found, get user data and predict cluster
                user_data = get_user(user_email)
                
                if user_data:
                    cluster = predict_cluster(user_data)
                    
                    if cluster is not None:
                        store_user_cluster(user_email, cluster)
                        cluster_value = cluster
                        logging.info(f"Predicted and stored cluster {cluster_value} for user {user_email}")
                    else:
                        logging.error(f"Failed to predict cluster for user {user_email}")
                        # Use default cluster based on gender
                        cluster_value = 1 if gender_str == "female" else 0
                else:
                    logging.error(f"User data not found for email: {user_email}")
                    return jsonify({"error": "User data not found"}), 404
            else:
                cluster_value = cluster_row['cluster']
                logging.info(f"Found existing cluster {cluster_value} for user {user_email}")
        except Exception as e:
            logging.error(f"Error retrieving/predicting user cluster: {str(e)}")
            # Use default cluster based on gender
            cluster_value = 1 if gender_str == "female" else 0
        finally:
            if conn:
                conn.close()
        
        # Get most popular products for the user's cluster from wishlist
        try:
            conn = get_amazon_db_connection()
            cursor = conn.cursor()
            # Count how many times each product (by ASIN) appears in the wishlist for this cluster
            cursor.execute("""
                SELECT 
                    asin, 
                    title, 
                    price, 
                    rating, 
                    reviews_count, 
                    image_url, 
                    product_url, 
                    is_prime, 
                    is_best_seller, 
                    is_amazon_choice,
                    COUNT(*) as popularity
                FROM wishlist
                WHERE cluster = ?
                GROUP BY asin
                ORDER BY popularity DESC
                LIMIT 30
            """, (cluster_value,))
            
            top_wishlist_products = [dict(row) for row in cursor.fetchall()]
            logging.info(f"Found {len(top_wishlist_products)} top wishlist products for cluster {cluster_value}")
        except Exception as e:
            logging.error(f"Error fetching top wishlist products: {str(e)}")
            top_wishlist_products = []
        finally:
            if conn:
                conn.close()
        
        # Get hardcoded product recommendations based on gender and cluster
        product_names = get_hardcoded_recommendations(gender_str, cluster_value)
        logging.info(f"Using hardcoded recommendations for {gender_str}, cluster {cluster_value}: {product_names}")
        
        # Get product details from database only (no API calls)
        products_details = {}
        
        for product_name in product_names:
            logging.info(f"Retrieving stored products for: {product_name}")
            
            # Get products from database
            stored_products = get_stored_products(product_name, limit=5)
            
            if stored_products:
                # We already have data for this product
                logging.info(f"Found {len(stored_products)} products in database for {product_name}")
                products_details[product_name] = stored_products
            else:
                # No products found for this query
                logging.warning(f"No stored products found for {product_name}")
                products_details[product_name] = []
        
        # Final response 
        final_response = {
            "recommendations": product_names,
            "product_details": products_details,
            "top_wishlist_products": top_wishlist_products  # Add top wishlist products to the response
        }
        
        logging.info(f"Returning recommendations with {len(product_names)} products and {len(top_wishlist_products)} top wishlist products")
        return jsonify(final_response)
            
    except Exception as e:
        logging.error(f"Server error in /get_products: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/products', methods=['GET'])
def get_all_products():
    """Get products from the database with optional filtering"""
    query = request.args.get('query', '')
    limit = int(request.args.get('limit', 200))
    offset = int(request.args.get('offset', 0))
    
    conn = get_amazon_db_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        if query:
            cursor.execute('''
            SELECT * FROM products 
            WHERE search_query LIKE ? OR title LIKE ?
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            ''', (f'%{query}%', f'%{query}%', limit, offset))
        else:
            cursor.execute('''
            SELECT * FROM products 
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            ''', (limit, offset))
            
        products = [dict(row) for row in cursor.fetchall()]
        
        # Get total count
        if query:
            cursor.execute('''
            SELECT COUNT(*) as count FROM products 
            WHERE search_query LIKE ? OR title LIKE ?
            ''', (f'%{query}%', f'%{query}%'))
        else:
            cursor.execute('SELECT COUNT(*) as count FROM products')
            
        count = cursor.fetchone()['count']
        
        return jsonify({
            'success': True,
            'products': products,
            'total': count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
    finally:
        conn.close()


        
@app.route('/search', methods=['GET'])
def handle_search():
    """Endpoint for product search using external API"""
    try:
        query = request.args.get('q', '').strip()
        logging.info(f"Search request received for query: {query}")

        if not query:
            return jsonify({"error": "Search query is required"}), 400

        params = {
            "query": query,
            "page": "1",
            "country": "US",
            "sort_by": "RELEVANCE",
            "product_condition": "ALL"
        }

        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        # Transform API response to match frontend expectations
        products = data.get('data', {}).get('products', [])
        formatted_products = []
        for p in products:
            # Clean price data
            price_str = p.get('product_price', '0')
            try:
                # Remove currency symbols and commas
                price = float(re.sub(r'[^\d.]', '', price_str))
            except:
                price = 0.0

            # Clean rating data
            rating_str = p.get('product_star_rating', '0')
            try:
                rating = float(rating_str)
            except:
                rating = 0.0
            # In the /search endpoint's formatted_products:
            formatted_products.append({
    'title': p.get('product_title', 'No Title'),
    'price': price,
    'rating': rating,
    'reviews_count': int(p.get('product_num_ratings', 0)),
    'image_url': p.get('product_photo', ''),
    'product_url': p.get('product_url', '#'),
    'is_prime': 'Prime' in p.get('product_delivery_message', ''),
    'is_best_seller': p.get('is_best_seller', False),
    'is_amazon_choice': p.get('is_amazon_choice', False),
    'asin': p.get('asin', '')  # Ensure ASIN is included
})

        return jsonify({
            "success": True,
            "products": formatted_products
        })

    except requests.exceptions.RequestException as e:
        logging.error(f"Search API error: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Search failed: {str(e)}"
        }), 500
    except Exception as e:
        logging.error(f"General search error: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
    

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)