from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError as SQLAlchemyIntegrityError

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

# --- App and Configuration ---

app = Flask(__name__)
CORS(app, supports_credentials=True)

# Get Database URL from environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set. Please check your .env file or deployment configuration.")

# Configure the main database connection
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Get RapidAPI configuration from environment variables
API_URL = "https://real-time-amazon-data.p.rapidapi.com/search"
HEADERS = {
    "x-rapidapi-key": os.environ.get("RAPIDAPI_KEY", ""),
    "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"
}

# Configuration
logging.basicConfig(level=logging.INFO)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model artifacts
try:
    # NOTE: Model artifacts (pkl files) must be available in the BASE_DIR
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
    kmeans_model = joblib.load(os.path.join(BASE_DIR, 'kmeans_model.pkl'))
    training_columns = joblib.load(os.path.join(BASE_DIR, 'training_columns.pkl'))
    numeric_cols = joblib.load(os.path.join(BASE_DIR, 'numeric_cols.pkl'))
    
    if "Cluster" in training_columns:
        training_columns.remove("Cluster")
except FileNotFoundError as e:
    logging.error(f"Model file not found: {e}. Cluster prediction will fail.")
except Exception as e:
     logging.error(f"Error loading model artifacts: {e}")

# --- SQLAlchemy Models (Replaces all CREATE TABLE SQL) ---

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.Text, nullable=False)
    email = db.Column(db.Text, unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.Text)
    location = db.Column(db.Text)
    shopping_frequency = db.Column(db.Text)
    annual_income = db.Column(db.Float)
    password = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class UserCluster(db.Model):
    __tablename__ = 'user_cluster'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.Text, unique=True, nullable=False)
    cluster = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class Product(db.Model):
    __tablename__ = 'products'
    id = db.Column(db.Integer, primary_key=True)
    asin = db.Column(db.Text, unique=True)
    title = db.Column(db.Text)
    price = db.Column(db.Float)
    original_price = db.Column(db.Float)
    currency = db.Column(db.Text)
    rating = db.Column(db.Float)
    reviews_count = db.Column(db.Integer)
    image_url = db.Column(db.Text)
    product_url = db.Column(db.Text)
    is_prime = db.Column(db.Boolean)
    is_best_seller = db.Column(db.Boolean)
    is_amazon_choice = db.Column(db.Boolean)
    climate_pledge_friendly = db.Column(db.Boolean)
    num_offers = db.Column(db.Integer)
    minimum_offer_price = db.Column(db.Text)
    sales_volume = db.Column(db.Text)
    delivery = db.Column(db.Text)
    has_variations = db.Column(db.Boolean)
    search_query = db.Column(db.Text)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)

class Wishlist(db.Model):
    __tablename__ = 'wishlist'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.Text, nullable=False)
    asin = db.Column(db.Text, nullable=False)
    cluster = db.Column(db.Integer, nullable=False)
    title = db.Column(db.Text, nullable=False)
    price = db.Column(db.Float, nullable=False)
    rating = db.Column(db.Float)
    reviews_count = db.Column(db.Integer)
    image_url = db.Column(db.Text)
    product_url = db.Column(db.Text)
    is_prime = db.Column(db.Boolean)
    is_best_seller = db.Column(db.Boolean)
    is_amazon_choice = db.Column(db.Boolean)
    created_at = db.Column(db.TIMESTAMP, default=datetime.utcnow)
    
    # Define a unique constraint on (email, asin)
    __table_args__ = (db.UniqueConstraint('email', 'asin', name='_email_asin_uc'),)


# Initialize Database (SQLAlchemy handles table creation based on Models)
with app.app_context():
    db.create_all()
    logging.info("PostgreSQL database tables initialized successfully via SQLAlchemy.")


# --- Database Helpers (Now using SQLAlchemy) ---

def get_user(email):
    """Fetch user data for processing/signin"""
    user = User.query.filter_by(email=email).first()
    
    if user:
        # Format data to match model training column names
        user_data = {
            'email': user.email,
            'password': user.password,
            'Age': user.age,
            'Gender': user.gender,
            'Income': user.annual_income,
            'Location': user.location,
            'Frequency of Purchases': user.shopping_frequency
        }
        # Use a dummy object mimicking sqlite3.Row for backward compatibility in signin/preprocess
        class RowMock(dict):
            def __getitem__(self, key):
                return self.get(key)
        
        return RowMock(user_data)
    return None

def store_user_cluster(email, cluster):
    """Store or update user cluster in database"""
    try:
        cluster_int = int(cluster)
        user_cluster = UserCluster.query.filter_by(email=email).first()
        
        if user_cluster:
            user_cluster.cluster = cluster_int
        else:
            new_cluster = UserCluster(email=email, cluster=cluster_int)
            db.session.add(new_cluster)
            
        db.session.commit()
    except Exception as e:
        logging.error(f"Error storing user cluster: {e}")
        db.session.rollback()

def get_stored_products(search_query=None, limit=200):
    """Function for product retrieval from existing database"""
    logging.info(f"Retrieving stored products for query: {search_query}")
    
    try:
        query = Product.query
        
        if search_query:
            # Use ilike for case-insensitive search in PostgreSQL
            query = query.filter(db.or_(
                Product.title.ilike(f'%{search_query}%'),
                Product.search_query.ilike(f'%{search_query}%')
            ))
            
        products = query.order_by(Product.created_at.desc()).limit(limit).all()
        
        # Convert SQLAlchemy objects to dicts
        products_dict = [{c.name: getattr(p, c.name) for c in p.__table__.columns} for p in products]
        
        logging.info(f"Retrieved {len(products_dict)} products from database")
        return products_dict
        
    except Exception as e:
        logging.error(f"Error in get_stored_products: {str(e)}")
        db.session.rollback()
        return []

def store_product_in_db(product_data, search_query):
    """Insert or replace a single product in the products table using Postgres ON CONFLICT."""
    try:
        product_instance = Product.query.filter_by(asin=product_data.get('asin')).first()
        
        if product_instance:
            # Update existing product
            product_instance.title = product_data.get('title')
            product_instance.price = product_data.get('price')
            product_instance.rating = product_data.get('rating')
            product_instance.reviews_count = product_data.get('reviews_count')
            product_instance.image_url = product_data.get('image_url')
            product_instance.product_url = product_data.get('product_url')
            product_instance.is_prime = product_data.get('is_prime')
            product_instance.is_best_seller = product_data.get('is_best_seller')
            product_instance.is_amazon_choice = product_data.get('is_amazon_choice')
            product_instance.search_query = search_query
            product_instance.created_at = datetime.utcnow()
        else:
            # Create new product
            new_product = Product(
                asin=product_data.get('asin'),
                title=product_data.get('title'),
                price=product_data.get('price'),
                rating=product_data.get('rating'),
                reviews_count=product_data.get('reviews_count'),
                image_url=product_data.get('image_url'),
                product_url=product_data.get('product_url'),
                is_prime=product_data.get('is_prime'),
                is_best_seller=product_data.get('is_best_seller'),
                is_amazon_choice=product_data.get('is_amazon_choice'),
                search_query=search_query
            )
            db.session.add(new_product)
            
        db.session.commit()
    except Exception as e:
        logging.error(f"Error storing product {product_data.get('asin')} in DB: {e}")
        db.session.rollback()


# --- Cluster Prediction Logic (No Change Needed) ---

def preprocess_user_data(user_row):
    """Transform user data to match model training format"""
    try:
        user_data = dict(user_row)

        for key in ["email", "password"]:
            if key in user_data:
                del user_data[key]

        df = pd.DataFrame([user_data])
        
        # Consistent mapping for gender
        df['Gender'] = df['Gender'].replace({0: 'female', 1: 'male', '0': 'female', '1': 'male'})
        
        df_encoded = pd.get_dummies(df, 
                                     columns=['Location', 'Frequency of Purchases', 'Gender'], 
                                     drop_first=True)
        
        missing_cols = set(training_columns) - set(df_encoded.columns)
        
        for col in missing_cols:
            df_encoded[col] = 0
        df_encoded = df_encoded[training_columns]
        
        for col in numeric_cols:
             if col in df_encoded.columns:
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce').fillna(0)
        
        df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])
        
        return df_encoded
    except Exception as e:
        logging.error(f"Error preprocessing user data: {e}")
        return None

def predict_cluster(user_data):
    """Predict cluster for preprocessed data"""
    try:
        processed_data = preprocess_user_data(user_data)
        if processed_data is not None:
            cluster = kmeans_model.predict(processed_data)[0]
            return cluster
    except Exception as e:
        logging.error(f"Error predicting cluster: {e}")
    return None

def get_hardcoded_recommendations(gender, cluster):
    """Return hardcoded product recommendations based on gender and cluster"""
    male_recommendations = {
        0: ["jewelry for male", "pants for male", "boots for male", "sunglasses for male", "shorts for male"],
        1: ["watches for male", "shoes for male", "jackets for male", "backpacks for male", "hats for male"], 
        2: ["shirt for male", "sweater for male", "belt for male", "pants for male", "gloves for male"],
        3: ["sneakers for male", "jeans for male", "hoodies for male", "wallets for male", "sunglasses for male"] 
    }
    female_recommendations = {
        0: ["dresses for female", "earrings for female", "scarves for female", "bracelets for female", "leggings for female"],
        1: ["sunglasses for female", "blouse for female", "boots for female", "socks for female", "shirt for female"],
        2: ["skirts for female", "dresses for female", "necklaces for female", "perfume for female", "makeup for female"],
        3: ["sandals for female", "handbag for female", "blouse for female", "belt for female", "shirt for female"]
    }
    gender = str(gender).lower()
    if gender in ['0', 'f', 'female']:
        gender_str = "female"
    else:
        gender_str = "male"
    
    if gender_str == "male":
        return male_recommendations.get(cluster, male_recommendations[0]) 
    else: 
        return female_recommendations.get(cluster, female_recommendations[1]) 


# --- Routes (Updated to use SQLAlchemy Models) ---

@app.route('/')
def serve_frontpage():
    return send_from_directory(BASE_DIR, 'frontpage.html')

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        password = bcrypt.hashpw(data['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        new_user = User(
            full_name=data['full_name'], 
            email=data['email'], 
            age=data['age'], 
            gender=data['gender'], 
            location=data['location'], 
            shopping_frequency=data['shopping_frequency'], 
            annual_income=data['annual_income'], 
            password=password
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({"message": "Signup Successful!"}), 201
    except SQLAlchemyIntegrityError:
        db.session.rollback()
        return jsonify({"error": "Email already registered!"}), 400
    except Exception as e:
        db.session.rollback()
        logging.error(f"Signup error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/signin_page.html')
def serve_signin_page():
    return send_from_directory(BASE_DIR, 'signin_page.html')

@app.route('/signin', methods=['POST'])
def signin():
    try:
        data = request.get_json()
        email = data.get("email")
        password = data.get("password")

        if not email or not password:
            return jsonify({"error": "Email and password are required"}), 400

        user_db = User.query.filter_by(email=email).first()

        if user_db and bcrypt.checkpw(password.encode('utf-8'), user_db.password.encode('utf-8')):
            user_data = get_user(email)
            
            cluster_row = UserCluster.query.filter_by(email=email).first()
            
            if not cluster_row:
                cluster = predict_cluster(user_data)
                if cluster is not None:
                    store_user_cluster(email, cluster)
            
            return jsonify({
                "message": "Signin successful!",
                "redirect": "product_page.html",
                "email": email
            })
        else:
            return jsonify({"error": "Invalid email or password"}), 401

    except Exception as e:
        logging.error(f"Signin error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/product_page.html')
def serve_product_page():
    return send_from_directory(BASE_DIR, 'product_page.html')

@app.route('/get_products', methods=['GET'])
def get_products():
    """Endpoint for fetching product recommendations from database only"""
    try:
        user_email = request.args.get('email')
        if not user_email:
            return jsonify({"error": "Email parameter is required"}), 400

        user_data = get_user(user_email)
        if not user_data:
            return jsonify({"error": "User data not found"}), 404

        user_gender = user_data['Gender']
        gender_str = "female" if str(user_gender).lower() in ['0', 'f', 'female'] else "male"
        
        cluster_value = None
        cluster_row = UserCluster.query.filter_by(email=user_email).first()
        
        if cluster_row:
            cluster_value = cluster_row.cluster
        else:
            cluster = predict_cluster(user_data)
            if cluster is not None:
                store_user_cluster(user_email, cluster)
                cluster_value = cluster
            else:
                cluster_value = 1 if gender_str == "female" else 0
                logging.warning(f"Failed to predict cluster, using default {cluster_value}")

        # Get most popular products for the user's cluster from wishlist
        top_wishlist_products_query = db.session.execute(text("""
            SELECT 
                asin, title, price, rating, reviews_count, image_url, product_url, 
                is_prime, is_best_seller, is_amazon_choice,
                COUNT(asin) AS popularity
            FROM wishlist
            WHERE cluster = :cluster_val
            GROUP BY 
                asin, title, price, rating, reviews_count, image_url, product_url, 
                is_prime, is_best_seller, is_amazon_choice
            ORDER BY popularity DESC
            LIMIT 30
        """), {'cluster_val': cluster_value})

        # Fetch the results as dicts
        top_wishlist_products = [dict(row) for row in top_wishlist_products_query.mappings().all()]
        
        product_names = get_hardcoded_recommendations(gender_str, cluster_value)
        products_details = {}
        
        for product_name in product_names:
            stored_products = get_stored_products(product_name, limit=5)
            products_details[product_name] = stored_products
        
        final_response = {
            "recommendations": product_names,
            "product_details": products_details,
            "top_wishlist_products": top_wishlist_products
        }
        
        return jsonify(final_response)
            
    except Exception as e:
        logging.error(f"Server error in /get_products: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/add_to_wishlist', methods=['POST'])
def add_to_wishlist():
    try:
        data = request.json
        email = data.get('email')
        cluster = data.get('cluster')
        product = data.get('product')
        
        if not email or cluster is None or not product or not product.get('asin'):
            return jsonify({"error": "Email, cluster, and product details (including ASIN) are required"}), 400

        new_wishlist_item = Wishlist(
            email=email, 
            cluster=int(cluster), 
            asin=product['asin'], 
            title=product.get('title', 'No Title'), 
            price=product.get('price', 0.0), 
            rating=product.get('rating'),
            reviews_count=product.get('reviews_count'), 
            image_url=product.get('image_url'), 
            product_url=product.get('product_url'),
            is_prime=product.get('is_prime'), 
            is_best_seller=product.get('is_best_seller'), 
            is_amazon_choice=product.get('is_amazon_choice')
        )
        
        db.session.add(new_wishlist_item)
        db.session.commit()
        
        return jsonify({"message": "Product added to wishlist"}), 201
    except SQLAlchemyIntegrityError:
        db.session.rollback()
        return jsonify({"error": "Product already in wishlist"}), 409
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error adding to wishlist: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500
    
@app.route('/remove_from_wishlist', methods=['POST'])
def remove_from_wishlist():
    try:
        data = request.json
        email = data.get('email')
        product_id = data.get('product_id')
        
        if not email or not product_id:
            return jsonify({"error": "Email and product_id are required"}), 400

        item = Wishlist.query.filter_by(email=email, id=product_id).first()
        
        if not item:
            return jsonify({"error": "Product not found in wishlist"}), 404
            
        db.session.delete(item)
        db.session.commit()

        return jsonify({"message": "Product removed from wishlist"}), 200

    except Exception as e:
        db.session.rollback()
        logging.error(f"Error removing from wishlist: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/get_wishlist', methods=['GET'])
def get_wishlist():
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        wishlist_items = Wishlist.query.filter_by(email=email).all()
        
        # Convert SQLAlchemy objects to dicts
        wishlist_products = [{c.name: getattr(item, c.name) for c in item.__table__.columns} for item in wishlist_items]

        return jsonify({"products": wishlist_products}), 200

    except Exception as e:
        logging.error(f"Error retrieving wishlist: {traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/get_user_cluster', methods=['GET'])
def get_user_cluster():
    try:
        email = request.args.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        cluster_row = UserCluster.query.filter_by(email=email).first()

        if not cluster_row:
            return jsonify({"error": "User cluster not found"}), 404

        return jsonify({"cluster": cluster_row.cluster}), 200

    except Exception as e:
        logging.error(f"Error fetching user cluster: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Other routes like get_category_products, api/products, search remain functionally the same, 
# but internally call the updated SQLAlchemy functions (get_stored_products, etc.)

@app.route('/get_category_products', methods=['GET'])
def get_category_products():
    try:
        user_email = request.args.get('email')
        category = request.args.get('category')
        
        if not user_email or not category:
            return jsonify({"error": "Email and category parameters are required"}), 400

        stored_products = get_stored_products(category, limit=50) 
        
        return jsonify({"products": stored_products})
            
    except Exception as e:
        logging.error(f"Server error in /get_category_products: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/products', methods=['GET'])
def get_all_products():
    try:
        query_text = request.args.get('query', '')
        limit = int(request.args.get('limit', 200))
        offset = int(request.args.get('offset', 0))
        
        # Base query
        base_query = Product.query
        
        if query_text:
            base_query = base_query.filter(db.or_(
                Product.search_query.ilike(f'%{query_text}%'),
                Product.title.ilike(f'%{query_text}%')
            ))

        # Get total count
        count = base_query.count()

        # Get paginated products
        products = base_query.order_by(Product.created_at.desc()).limit(limit).offset(offset).all()
        
        products_dict = [{c.name: getattr(p, c.name) for c in p.__table__.columns} for p in products]
        
        return jsonify({
            'success': True,
            'products': products_dict,
            'total': count,
            'limit': limit,
            'offset': offset
        })
        
    except Exception as e:
        logging.error(f"Error in /api/products: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        }), 500
    
@app.route('/search', methods=['GET'])
def handle_search():
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
    
    formatted_products = []
    try:
        response = requests.get(API_URL, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        products = data.get('data', {}).get('products', [])
        
        for p in products:
            asin = p.get('asin', '')
            if not asin:
                 continue
                 
            price_str = p.get('product_price', '$0.00')
            try:
                price = float(re.sub(r'[^\d.]', '', price_str))
            except:
                price = 0.0

            rating_str = p.get('product_star_rating', '0')
            try:
                rating = float(rating_str)
            except:
                rating = 0.0
            
            product_entry = {
                'title': p.get('product_title', 'No Title'),
                'price': price,
                'rating': rating,
                'reviews_count': int(p.get('product_num_ratings', 0)),
                'image_url': p.get('product_photo', ''),
                'product_url': p.get('product_url', '#'),
                'is_prime': 'Prime' in p.get('product_delivery_message', ''),
                'is_best_seller': p.get('is_best_seller', False),
                'is_amazon_choice': p.get('is_amazon_choice', False),
                'asin': asin,
            }
            formatted_products.append(product_entry)
            
            store_product_in_db(product_entry, query)

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
        logging.error(f"General search error: {traceback.format_exc()}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500
    
@app.route('/wishlist.html')
def serve_wishlist_page():
    return send_from_directory(BASE_DIR, 'wishlist.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
