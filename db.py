import psycopg2

CONN_STRING = "host='localhost' dbname='diplomovka' user='postgres'"

'''
TODO simplify query
'''


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
            and date < '2015-01-01 00:00:00'
         ), p_time_observations AS (
        SELECT
        f.validity_date,
        f.reference_date,
        o.humidity as p_time_humidity,
        o.wind_direction as p_time_wind_direction,
        o.pressure as p_time_pressure,
        o.wind_speed as p_time_wind_speed,
        o.rainfall_last_hour as p_time_rainfall_last_hour,
        o.temperature as p_time_temp
        FROM observations o, forecast f
        WHERE o.wheather_station_id = {0} AND
            o.wheather_station_id = f.wheather_station_id AND
            f.rr < 13 AND f.rr > 0 AND f.reference_date = o.date
            and reference_date < '2015-01-01 00:00:00'
        ), current_observations AS (
        SELECT
            o.temperature as current_temp,
            o.humidity as current_humidity,
            o.wind_direction as current_wind_direction,
            o.pressure as current_pressure,
            o.wind_speed as current_wind_speed,
            o.rainfall_last_hour as current_rainfall_last_hour,
            o.date as observation_date
        FROM observations o
            WHERE o.wheather_station_id = {0} and date < '2015-01-01 00:00:00'
        )
    SELECT
    c.reference_date,
    c.validity_date,
    c.p_time_temp,
    c.p_time_humidity,
    c.p_time_wind_direction,
    c.p_time_pressure,
    c.p_time_wind_speed,
    c.p_time_rainfall_last_hour,
    f.future_temp_shmu,
    f.future_temp,
    co.current_temp,
    co.current_humidity,
    co.current_wind_direction,
    co.current_pressure,
    co.current_wind_speed,
    co.current_rainfall_last_hour
    FROM p_time_observations c, future_temp f, current_observations co
    WHERE
        c.validity_date = f.validity_date and
        co.observation_date = c.validity_date - interval '1' hour
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
