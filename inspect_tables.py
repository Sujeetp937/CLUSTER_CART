import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect('cluster_cart.db')
cursor = conn.cursor()

# Inspect the structure of the user_cluster table
cursor.execute("PRAGMA table_info(user_cluster)")
print("user_cluster columns:")
for col in cursor.fetchall():
    print(col)

# Inspect the structure of the users table
cursor.execute("PRAGMA table_info(users)")
print("\nusers table columns:")
for col in cursor.fetchall():
    print(col)

conn.close()
