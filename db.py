import sqlite3
from datetime import datetime

def dict_factory(cursor, row):
    d = {}
    for idx,col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

db = sqlite3.connect("database.db", check_same_thread=False)
db.row_factory = dict_factory
cursor = db.cursor()

# Regions Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS regions (
    region_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    x_min INTEGER,
    y_min INTEGER,
    x_max INTEGER,
    y_max INTEGER
)
""")

# Logs Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id INTEGER,
    object_id INTEGER,
    active BOOLEAN,
    entry_time TEXT,
    dwell TEXT,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS density_event (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id INTEGER,
    active BOOLEAN,
    start_time TEXT,
    end_time TEXT,
    FOREIGN KEY (region_id) REFERENCES regions(region_id)
)
""")

db.commit()

def create_region(region_name, x_min, y_min, x_max, y_max):
    cursor.execute("""
    INSERT INTO regions (name, x_min, y_min, x_max, y_max)
    VALUES (?, ?, ?, ?, ?)
    """, (region_name, x_min, y_min, x_max, y_max))
    db.commit()
    print(f"Region created: {region_name}, Coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")

def get_regions():
    cursor.execute("""SELECT * FROM regions""")
    regions = cursor.fetchall()
    return regions

def get_region_count(region_name):
    cursor.execute("""
    SELECT COUNT(*) as count
    FROM logs
    WHERE region_id = (SELECT region_id FROM regions WHERE name = ?)
    AND active = 1
    """, (region_name,))
    
    count = cursor.fetchone()["count"]
    return count

def get_logs(region_name):
    cursor.execute("""
    WITH TimePoints AS (
        -- All entry times
        SELECT entry_time as timestamp, 1 as change
        FROM logs 
        WHERE region_id = (SELECT region_id FROM regions WHERE name = ?)
        
        UNION ALL
        
        -- Only completed exit times
        SELECT datetime(entry_time, '+' || dwell) as timestamp, -1 as change
        FROM logs 
        WHERE region_id = (SELECT region_id FROM regions WHERE name = ?)
        AND active = 0
        AND dwell IS NOT NULL
    )
    SELECT timestamp, SUM(change) OVER (ORDER BY timestamp) as count
    FROM TimePoints
    ORDER BY timestamp
    """, (region_name, region_name))
    return cursor.fetchall()
    
def get_active_density_event(region_name):
    cursor.execute("""
    SELECT * FROM density_event
    WHERE region_id = (SELECT region_id FROM regions WHERE name = ?)
    AND active = 1
    """, (region_name,))
    return cursor.fetchone()

def get_active_log(region_name, obj_id):
    cursor.execute("""
    SELECT * FROM logs
    WHERE region_id = (SELECT region_id FROM regions WHERE name = ?)
    AND object_id = ? AND active = 1
    """, (region_name, obj_id))
    return cursor.fetchone()

def update_log_event(region_name, obj_id):
    cursor.execute("""
    SELECT region_id FROM regions WHERE name = ?
    """, (region_name,))
    region = cursor.fetchone()

    if not region:
        print(f"Region '{region_name}' not found.")
        return

    region_id = region['region_id']

    cursor.execute("""
    SELECT log_id, entry_time FROM logs
    WHERE region_id = ? AND object_id = ? AND active = 1
    """, (region_id, obj_id))
    log = cursor.fetchone()

    if log:
        log_id = log['log_id']
        entry_time = datetime.strptime(log['entry_time'], "%Y-%m-%d %H:%M:%S")
        exit_time = datetime.now()
        dwell_time = (exit_time - entry_time).total_seconds()  # Store as seconds instead of string

        cursor.execute("""
        UPDATE logs
        SET active = 0, dwell = ?
        WHERE log_id = ?
        """, (dwell_time, log_id))

        print(f"Updated log for object {obj_id} in region '{region_name}'. Dwell time: {dwell_time}")
    else:
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO logs (region_id, object_id, active, entry_time, dwell)
        VALUES (?, ?, 1, ?, NULL)
        """, (region_id, obj_id, entry_time))

        print(f"Inserted new log for object {obj_id} in region '{region_name}' at {entry_time}.")

    db.commit()

def update_density_event(region_name):
    cursor.execute("""
    SELECT name, region_id FROM regions WHERE name = ?
    """, (region_name,))
    region = cursor.fetchone()

    if not region:
        print(f"Region '{region_name}' not found.")
        return

    region_id = region['region_id']

    cursor.execute("""
    SELECT * FROM density_event 
    WHERE region_id = ? AND active = 1
    """, (region_id,))
    event = cursor.fetchone()

    if event: 
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        UPDATE density_event
        SET active = 0, end_time = ?
        WHERE event_id = ?
        """, (end_time, event['event_id']))

        print(f"Updated density event for region '{region_name}'.")
        return {
            "zone": region["name"],
            "start_time": event["start_time"],
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    else:
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO density_event (region_id, active, start_time, end_time)
        VALUES (?, 1, ?, NULL)
        """, (region_id, start_time))

        print(f"Inserted new density event for region '{region_name}'")

    db.commit()
