import sqlite3

# Connect to the database
conn = sqlite3.connect('cluster_cart.db')
cursor = conn.cursor()

# Check columns of user_cluster table
cursor.execute("PRAGMA table_info(user_cluster)")
print("user_cluster columns:")
for col in cursor.fetchall():
    print(col)

# Check columns of users table (optional but helpful)
cursor.execute("PRAGMA table_info(users)")
print("\nusers table columns:")
for col in cursor.fetchall():
    print(col)

conn.close()
