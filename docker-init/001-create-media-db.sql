-- Create the application database used by the media-flow pipeline
CREATE DATABASE media_flow_db
  WITH OWNER = airflow
       ENCODING = 'UTF8'
       TEMPLATE = template0;

GRANT ALL PRIVILEGES ON DATABASE media_flow_db TO airflow;