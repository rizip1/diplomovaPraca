import csv
import psycopg2
import os

conn_string = "host='localhost' dbname='diplomovka' user='postgres'"
conn = psycopg2.connect(conn_string)
conn.autocommit = True

prefix = '../obs/'

for observation in os.listdir(prefix):
    with open(prefix + observation, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if (row):
                row.pop()
                row.append(observation.split('.')[0])
                cur = conn.cursor()
                query = """INSERT INTO observations (
                           date, temperature, humidity, pressure,
                           wind_direction, wind_speed, rainfall_last_hour,
                           wheather_station_id)
                           VALUES  ('{}',{},{},{},{},{},{},{})""".format(*row)
                cur.execute(query)

