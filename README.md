## Analysis code for high frequency locking in weakly electric fish

### Installation instructions

 - install docker
 - `sudo pip install docker-compose`
 - Clone code:  `git clone https://github.com/fabiansinz/efish_locking.git`
 - Create a file `.env` with content
 
    ```
    DJ_HOST=<IP OF MYSQL-SERVER>
    DJ_USER=<USER FOR THIS DATABASE>
    DJ_PASS=<PASSWORD FOR THE USER>
    ```
    
    and replace the placeholders with the correct values. It has to be the IP, because the Docker container cannot look up servers.
    The mysql user (here it's `efish`) needs the rights `GRANT ALL PRIVILEGES ON `efish%`.* TO 'efish'@'%';`
 - Build the docker container: `sudo docker-compose build --pull --no-cache locking`
 - If you only want to use a populated database, read on. If you want to populate the tables, then you'll need a directory
    `data/carolin` in you home with the recordings. If the files are at a different location, change the location in
    `docker-compose.yml`.
 - Start the docker container by `sudo docker-compose run locking`. That will start a shell in the docker container
 - To import the data use: `python3 scripts/populate_data.py`
 - To run analyses use: `python3 scripts/populate_analyses.py`
 - To run modells use: `python3 scripts/populate_modelling.py`
 - After that you can reproduce the figures with the respective figure scripts in the `scripts` directory.
   The current `docker-compose.yml` maps the local directory `figures_docker` to the directory where the
   figures are stored in the container. This means you either have to create that one locally or you need to
   change the `docker-compose.yml` to point to a different directory.
