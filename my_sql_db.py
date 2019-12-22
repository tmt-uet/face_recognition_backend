import mysql.connector


class Database:

    def __init__(self):
        self.connection = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd="",
            database="recognize"
        )

    def query(self, q, arg=()):
        cursor = self.connection.cursor()

        cursor.execute(q, arg)
        results = cursor.fetchall()
        cursor.close()

        return results

    def insert(self, q, arg=()):
        cursor = self.connection.cursor()

        cursor.execute(q, arg)

        self.connection.commit()
        result = cursor.lastrowid
        cursor.close()
        return result

    def select(self, q, arg=()):
        cursor = self.connection.cursor()
        cursor.execute(q, arg)
        records = cursor.fetchall()
        # return cursor.execute(q, arg)
        return records

    def delete(self, q, arg=()):
        cursor = self.connection.cursor()
        result = cursor.execute(q, arg)
        self.connection.commit()
        return result
