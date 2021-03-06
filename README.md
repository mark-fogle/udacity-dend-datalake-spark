# Project: Data Lake using Amazon Web Services S3 and Apache Spark

## Scenario

A music streaming startup, Sparkify, has grown their user base and song database even more and want to move their data warehouse to a data lake. Their data resides in S3, in a directory of JSON logs on user activity on the app, as well as a directory with JSON metadata on the songs in their app.

This project will build an ETL pipeline that extracts their data from S3, processes them using Spark, and loads the data back into S3 as a set of dimensional tables. This will allow their analytics team to continue finding insights in what songs their users are listening to.

## Project Datasets

This project uses two datasets from AWS S3:

* Song data: s3://udacity-dend/song-data
* Log data: s3://udacity-dend/log-data

### Song Dataset

The first dataset is a subset of real data from the [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/). Each file is in JSON format and contains metadata about a song and the artist of that song. The files are partitioned by the first three letters of each song's track ID.

### Log Dataset

The second dataset consists of log files in JSON format generated by this [event simulator](https://github.com/Interana/eventsim) based on the songs in the dataset above. These simulate app activity logs from an imaginary music streaming app based on configuration settings.

The log files in the dataset are partitioned by year and month.

## Project Files

The project template includes the following files:

* etl.py - loads data from S3, use Apache Spark to extract and then save to S3 as Parquet
* dl.cfg - configuration file for ETL process
* data/song-data.zip - Sample song dataset for development
* data/log-data.zip - Sample log dataset for development
* QueryTest.ipynb - Notebook for testing queries

## Schema

This project will transform the data from AWS S3 from JSON into Parquet creating a star schema with fact and dimension tables.

### Fact Table

songplays - records in event data associated with song plays i.e. records with page NextSong, partitioned by year then month
    songplay_id, start_time, user_id, level, song_id, artist_id, session_id, location, user_agent

### Dimension Tables

users - users in the app
    user_id, first_name, last_name, gender, level
songs - songs in music database partitioned by year then artist
    song_id, title, artist_id, year, duration
artists - artists in music database
    artist_id, name, location, lattitude, longitude
time - timestamps of records in songplays broken down into specific units, partitioned by year then month
    start_time, hour, day, week, month, year, weekday

## Deployment

Configure dl.cfg.
Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY with creedentials that have read write access to S3.
INPUT_PATH should be set to s3a://udacity-dend/ for Udacity data set
OUTPUT_PATH should be set to a S3 bucket to which you have access.

Copy dl.cfg and etl.py to your EMR cluster

```sh
scp -i <path to your SSH key> etl.py <your emr host>:~/
scp -i <path to your SSH key> dl.cfg <your emr host>:~/
```

## Execution

SSH to EMR cluster.

Submit Apache Spark job.

```sh
/usr/bin/spark-submit --master yarn etl.py
```
