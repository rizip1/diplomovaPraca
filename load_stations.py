import csv
import psycopg2


conn_string = "host='localhost' dbname='diplomovka' user='postgres'"
conn = psycopg2.connect(conn_string)
conn.autocommit = True

with open('stations.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
    for row in reader:
        if (row):
            row.pop()
            cur = conn.cursor()
            query = """INSERT INTO wheather_stations (wheather_station_id,
                       latitude, longitude, altitude, name)
                       VALUES  ({},{},{}, {}, '{}')""".format(*row)
            cur.execute(query)

