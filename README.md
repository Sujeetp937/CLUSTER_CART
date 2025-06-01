# ClusterCart: Smart Shopping Recommendations

ClusterCart is an intelligent shopping recommendation system that uses machine learning (K-means clustering) to personalize product recommendations based on user demographics and shopping behavior. The application analyzes user data to group them into clusters with similar characteristics and delivers tailored product suggestions from Amazon's marketplace.

## ✨ Features

- **User Authentication**: Secure signup and signin with password hashing
- **ML-Powered Recommendations**: Personalized product recommendations using K-means clustering
- **Amazon Product Integration**: Real-time product data from Amazon's marketplace
- **Wishlist Management**: Save and manage favorite products
- **Product Search**: Search Amazon's product catalog in real-time
- **Responsive UI**: User-friendly interface for desktop and mobile

## 🔧 Technology Stack

- **Backend**: Flask (Python)
- **Database**: SQLite
- **Machine Learning**: Scikit-learn (K-means clustering)
- **API Integration**: RapidAPI Amazon Data API
- **Authentication**: Bcrypt password hashing
- **Frontend**: HTML, CSS, JavaScript (Not included in this repo)

## 📋 Prerequisites

Before running the application, make sure you have:

- Python 3.8+ installed
- RapidAPI key for the "Real-Time Amazon Data" API
- Required Python packages (see Installation section)

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/shyam1902005/CLUSTER_CART.git
cd CLUSTER_CART
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

#### Activate virtual environment:

**Windows**:
```bash
venv\Scripts\activate
```

**macOS/Linux**:
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The requirements.txt file contains all necessary dependencies for the project.

### 4. Configure environment variables

Create a `.env` file in the project root directory with your RapidAPI key:

```
RAPIDAPI_KEY=your_rapidapi_key_here
```


To get a RapidAPI key:
1. Sign up at [RapidAPI](https://rapidapi.com/)
2. Subscribe to [Real-Time Amazon Data API](https://rapidapi.com/letscrape-6bRBa3QguO5/api/real-time-amazon-data/)
3. Copy your API key from your dashboard

### 5. Prepare ML models

Make sure the following model files are in the project root directory:
- `scaler.pkl` - Standard scaler for numeric features
- `kmeans_model.pkl` - Trained K-means clustering model
- `training_columns.pkl` - Features used in training
- `numeric_cols.pkl` - Numeric columns for scaling

If these files are missing, you'll need to train the models before running the application.

### 6. Run the application

```bash
python app.py
```

The server will start on `http://localhost:5000`.

## 🗄️ Database Structure

The application uses two SQLite databases:

### 1. cluster_cart.db

**Users Table:**
- User demographics and authentication data

**User_Cluster Table:**
- Maps users to their assigned clusters

### 2. amazon_products.db

**Products Table:**
- Cached Amazon product data

**Wishlist Table:**
- User wishlist items

## 🔄 API Endpoints

### Authentication Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/signup` | POST | Register a new user |
| `/signin` | POST | Authenticate a user |

### Product Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_products` | GET | Get personalized product recommendations |
| `/get_category_products` | GET | Get products for a specific category |
| `/search` | GET | Search for products |
| `/api/products` | GET | Get all products with optional filtering |

### Wishlist Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/add_to_wishlist` | POST | Add a product to wishlist |
| `/get_wishlist` | GET | Get user's wishlist |
| `/remove_from_wishlist` | POST | Remove a product from wishlist |

### User Data Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/get_user_cluster` | GET | Get the user's assigned cluster |

### Static Pages

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve front page |
| `/signin_page.html` | GET | Serve signin page |
| `/product_page.html` | GET | Serve product page |
| `/wishlist.html` | GET | Serve wishlist page |

## 🛠️ How It Works

1. **User Registration & Data Collection**:
   - Upon registration, the application collects demographic data such as age, gender, location, shopping frequency, and annual income.

2. **Cluster Assignment**:
   - The K-means model analyzes user data and assigns the user to one of four clusters of similar shoppers.

3. **Personalized Recommendations**:
   - Based on the assigned cluster and gender, the system recommends relevant product categories.
   - Products are retrieved from either the database cache or real-time from Amazon's API.

4. **Wishlist Learning**:
   - The system learns from cluster-wide wishlist patterns to further refine recommendations.

## 📊 Machine Learning Model

ClusterCart uses K-means clustering with the following features:
- Age
- Gender
- Income
- Location (one-hot encoded)
- Shopping Frequency (one-hot encoded)

## 🔐 Security Features

- Password hashing with bcrypt
- Input validation
- Error handling
- CORS support for front-end integration

## 🧪 Testing

To test the API endpoints, you can use tools like Postman or curl:

### Example: Testing the signin endpoint

```bash
curl -X POST http://localhost:5000/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com", "password":"your_password"}'
```

## 📝 Development Notes

- The application uses SQLite for simplicity, but can be scaled with PostgreSQL or MySQL for production.
- RapidAPI calls are limited based on your subscription plan.
- For production, consider implementing proper rate limiting and caching strategies.

## 🚧 Future Improvements

- Add unit and integration tests
- Implement user feedback system for recommendations
- Enhance the recommendation algorithm with collaborative filtering
- Add product comparison features
- Integrate payment processing for a complete e-commerce solution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

If you have any questions or feedback, please reach out at:

- Email: sujeetpal2606@gmail.com
- GitHub: [Sujeetp937](https://github.com/Sujeetp937)

---

Made with ❤️ by [Sujeet Pal]
