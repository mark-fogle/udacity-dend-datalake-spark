import configparser
import datetime
import os
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format
from pyspark.sql.types import StructType as R, StructField as Fld, DoubleType as Dbl, StringType as Str, IntegerType as Int, DateType as Date, TimestampType
from pyspark.sql.functions import monotonically_increasing_id

config = configparser.ConfigParser()
config.read_file(open('dl.cfg'))

os.environ["AWS_ACCESS_KEY_ID"]= config['AWS']['AWS_ACCESS_KEY_ID']
os.environ["AWS_SECRET_ACCESS_KEY"]= config['AWS']['AWS_SECRET_ACCESS_KEY']

INPUT_PATH=config['AWS']['INPUT_PATH']
OUTPUT_PATH=config['AWS']['OUTPUT_PATH']

def create_spark_session():
    """
    Creates Spark session
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    
    spark.conf.set("mapreduce.fileoutputcommitter.algorithm.version", "2")
    
    return spark


def process_song_data(spark, input_data, output_data):
    """
    Loads song data from base S3 location 
    and extracts JSON data and stores as Parquet
    partitioned by year and artist_id.
    
    Parameters
    ----------
    spark : SparkSession
        Apache Spark session
    input_data : str
        The path prefix for the song data.
    output_data : str
        The path prefix for the output data.
    
    """
    
    # get filepath to song data file
    song_data = "{}song-data/*/*/*/*.json".format(input_data)
        
    #create song data schema
    songSchema = R([
        Fld("num_songs", Int()),
        Fld("artist_id", Str()),
        Fld("artist_latitude", Str()),
        Fld("artist_longitude", Str()),
        Fld("artist_location", Str()),
        Fld("artist_name", Str()),
        Fld("song_id", Str()),
        Fld("title", Str()),
        Fld("duration", Dbl()),
        Fld("year", Int()),
    ])

    # read song data file
    df = spark.read.json(song_data, schema = songSchema)
    
    # extract songs table based on unique song_id
    songs_table = df.select(df.song_id, \
                            df.title, \
                            df.artist_id, \
                            df.year, \
                            df.duration) \
                    .dropDuplicates(["song_id"])
    
    # create songs view
    songs_table.createOrReplaceTempView('songs')

    # write songs table to parquet files partitioned by year and artist
    songs_table.write.mode("overwrite").partitionBy("year", "artist_id").parquet("{}songs".format(output_data))

    # create artists table based on unique artist_id
    artists_table = df.select(df.artist_id, \
                              df.artist_name.alias("name"), \
                              df.artist_location.alias("location"), \
                              df.artist_latitude.alias("lattitude"), \
                              df.artist_longitude.alias("longitude")) \
                      .dropDuplicates(["artist_id"])
    
    # write artists table to parquet files
    artists_table.write.mode("overwrite").parquet("{}/artists".format(output_data))


def process_log_data(spark, input_data, output_data):
    """
    Loads log event data from base S3 location,
    extracts JSON data.
    Creates users table and stores as Parquet.
    Creates time table and stores as Parquet
    partitioned by year and month.
    Creates songplays table and stores as Parquet
    partitioned by year and month.
    
    Parameters
    ----------
    spark : SparkSession
        Apache Spark session
    input_data : str
        The path prefix for the song data.
    output_data : str
        The path prefix for the output data.
    
    """
    
    # get filepath to log data file
    log_data = "{}log-data/*/*/*.json".format(input_data)

    # read log data file
    df = spark.read.json(log_data)
           
    # filter by actions for song plays
    df = df.filter(df['page'] == "NextSong")
    
    # create primary key
    df = df.withColumn("songplay_id", monotonically_increasing_id()+1)
    df.createOrReplaceTempView("logs")
    
    users_table = df.select(df.userId.alias("user_id"), \
                           df.firstName.alias("first_name"), \
                           df.lastName.alias("last_name"), \
                           df.gender, \
                           df.level).dropDuplicates(["user_id"])
    
    # write users table to parquet files
    users_table.write.mode("overwrite").parquet("{}users".format(output_data))
    
    # create timestamp column from Unix milliseconds timestamp "ts"
    
    # define UDF for extracting timestamp
    get_timestamp = udf(lambda x: datetime.datetime.fromtimestamp(x / 1000), TimestampType())
    
    # create time dataframe, drop duplicates and any NA
    time_df = df.select(df.ts.alias("start_time")) \
                .dropDuplicates(["start_time"]) \
                .dropna()

    # extract timestamp column
    time_df = time_df.withColumn("timestamp", get_timestamp(time_df["start_time"]))
    
    # create hour, day, week, month, year, and week day columns
    time_df = time_df.withColumn('hour', F.hour(time_df.timestamp)) \
                     .withColumn('day', F.dayofmonth(time_df.timestamp)) \
                     .withColumn('week', F.weekofyear(time_df.timestamp)) \
                     .withColumn('month', F.month(time_df.timestamp)) \
                     .withColumn('year', F.year(time_df.timestamp)) \
                     .withColumn('weekday', F.dayofweek(time_df.timestamp)) \
    
    # select final columns and order by start_time
    time_df = time_df.select(time_df.start_time, time_df.hour, time_df.day, time_df.week, time_df.month, time_df.year, time_df.weekday) \
                     .sort("start_time")
    
    # create time_table view
    time_df.createOrReplaceTempView("time_table")
    
    # write time table to parquet files partitioned by year and month
    time_df.write.mode("overwrite").partitionBy("year", "month").parquet("{}time".format(output_data))

    # extract columns from joined song and log datasets to create songplays table 
    songplays_table = spark.sql("""
        SELECT l.songplay_id, l.ts as start_time, 
        l.userId as user_id, l.level, s.song_id, s.artist_id, 
        l.sessionId as session_id, l.location, l.userAgent as user_agent, t.year, t.month
        FROM logs l 
        JOIN songs s ON s.title=l.song
        JOIN time_table t ON l.ts=t.start_time
        """)

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write.mode("overwrite").partitionBy("year", "month").parquet("{}songplays".format(output_data))


def main():
    spark = create_spark_session()
    input_data = INPUT_PATH
    output_data = OUTPUT_PATH
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
