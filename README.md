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
