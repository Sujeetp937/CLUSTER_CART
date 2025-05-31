import sqlite3

# Connect to the database
conn = sqlite3.connect('cluster_cart.db')
cursor = conn.cursor()

# Corrected query: join on email
query = """
SELECT users.email, users.gender, user_cluster.cluster
FROM users
JOIN user_cluster ON users.email = user_cluster.email
"""

cursor.execute(query)
rows = cursor.fetchall()

# Display results
print("User Email\t\tGender\tCluster")
print("-" * 40)
for row in rows:
    print(f"{row[0]:<24}{row[1]:<8}{row[2]}")

conn.close()
