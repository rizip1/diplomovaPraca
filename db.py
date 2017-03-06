import psycopg2

CONN_STRING = "host='localhost' dbname='diplomovka' user='postgres'"


def save_data_for_station(station_id, out_file):
    query = '''
    WITH future_temp AS (
        SELECT
            o.temperature as future_temp, f.validity_date,
            f.temperature as future_temp_shmu
        FROM observations o, forecast f
        WHERE o.wheather_station_id = {0} AND
            o.wheather_station_id = f.wheather_station_id AND
            f.rr < 13 AND f.rr > 0 AND f.validity_date = o.date
         ), current_temp AS (
        SELECT o.temperature as current_temp, f.validity_date,
            f.reference_date, o.humidity, o.wind_direction, o.pressure,
            o.wind_speed, o.rainfall_last_hour
        FROM observations o, forecast f
        WHERE o.wheather_station_id = {0} AND
            o.wheather_station_id = f.wheather_station_id AND
            f.rr < 13 AND f.rr > 0 AND f.reference_date = o.date
        )
    SELECT c.reference_date, c.validity_date, c.humidity, c.wind_direction,
        c.current_temp, c.pressure, c.wind_speed, c.rainfall_last_hour,
        f.future_temp_shmu, f.future_temp
    FROM current_temp c, future_temp f
    WHERE c.validity_date = f.validity_date
    ORDER BY c.reference_date, c.validity_date
    '''.format(station_id)

    outputquery = "COPY ({0}) TO STDOUT WITH CSV HEADER DELIMITER ';'".format(
        query)

    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    with open(out_file, 'w') as f:
        cur.copy_expert(outputquery, f)

    conn.close()


def get_stations():
    query = '''
        SELECT wheather_station_id
        FROM wheather_stations
        ORDER BY wheather_station_id
    '''
    conn = psycopg2.connect(CONN_STRING)
    cur = conn.cursor()
    cur.execute(query)

    records = []

    for record in cur:
        records.append(int(record[0]))

    # Problematic station (check later)
    records.remove(11841)

    # Empty station (check later)
    records.remove(11955)

    conn.close()
    return records
