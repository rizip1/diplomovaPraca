import csv
import psycopg2
import os
import math

conn_string = "host='localhost' dbname='diplomovka' user='postgres'"
conn = psycopg2.connect(conn_string)
conn.autocommit = True

prefix = '../forecast/'

for f in os.listdir(prefix):
    with open(prefix + f, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in reader:
            if (row):
                row.pop()
                                
                data = {}

                data['reference_date'] = "{}-{}-{} {}:00:00".format(f[0:4],
                        f[4:6], f[6:8], f[9:11])
                data['wheather_station_id'] = row[0]
                data['rr'] = row[1]
                data['validity_date'] = row[2]
                data['altitude'] = row[3]
                data['temperature'] = row[4]
                data['humidity'] = row[5]
                data['pressure'] = row[6]
                data['cloudiness_total'] = row[7]
                data['cloudiness_low'] = row[8]
                data['rainfall_total'] = row[9]
                data['snow_rainfall_k'] = float(row[9]) / 100
                data['snow_rainfall_s'] = float(row[10]) / 100
                data['snow_rainfall'] = data['snow_rainfall_k'] + \
                        data['snow_rainfall_s']
                data['wind_u'] = float(row[11])
                data['wind_v'] = float(row[12])
                
                u = data['wind_u']
                v = data['wind_v']
                
                data['wind_speed'] = math.sqrt(u**2 + v**2)

                dd = 180.0 * math.atan2(u, v) / math.pi
                if (dd < 0):
                    dd = 180 + 180.0 * math.atan2(u, v) / math.pi
                data['wind_direction'] = dd
                
                cur = conn.cursor()
                query = """
                    INSERT INTO forecast (
                    wheather_station_id, rr, validity_date,
                    reference_date, altitude, temperature,
                    humidity, pressure, cloudiness_total,
                    cloudiness_low, rainfall_total, snow_rainfall_k,
                    snow_rainfall_s, snow_rainfall, wind_u, wind_v,
                    wind_speed, wind_direction)
                    VALUES  ({wheather_station_id},{rr},'{validity_date}',
                    '{reference_date}',{altitude},{temperature},{humidity},
                    {pressure},{cloudiness_total},{cloudiness_low},
                    {rainfall_total},{snow_rainfall_k},{snow_rainfall_s},
                    {snow_rainfall},{wind_u},{wind_v},{wind_speed},
                    {wind_direction})""".format(**data)
                cur.execute(query)
print('Data loaded')

