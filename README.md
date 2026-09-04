# parser

comandi per far partire il backend
> cd backend/src

> uvicorn server:app --reload --port 8003

comandi per far partire il fronend
> cd frontend/src

> uvicorn frontend:app --reload --port 8004

pagina web
http://localhost:8004/





LINUX--------------
basta fare 
    docker compose up --build

e attendere (un bel po') che scarichi tutto: si creeranno le cartelle mariadb_data e ollama_data che nei push/pull vengono ignorati quindi tranquilli

---------------------------------------------------------------
per verificare il giudizio dell'llm (ancora da perfezionare):
    far partire tutto con   docker compose up --build
    aprire un altro terminale e scrivere prompt del tipo:

            curl -X POST http://localhost:8003/evaluate_judge \
  -             H "Content-Type: application/json" \
                -d '{
                    "parsed_text": "Adversarial machine learning is the study of attacks on machine learning algorithms and defenses against such attacks. Most common attacks include evasion attacks, data poisoning attacks, Byzantine attacks and model extraction.",

                    "gold_text": "Adversarial machine learning is the study of the attacks on machine learning algorithms, and of the defenses against such attacks. Most common attacks in adversarial machine learning include evasion attacks, data poisoning attacks, Byzantine attacks and model extraction."
                }'

    e aspettate (un bel po') per i risultati (score + giudizio). Al momento è l'unico modo comodo per controllare

----------------------------------------------------------------------
per verificare invece il database:
    ar partire tutto con   docker compose up --build
    aprire un altro terminale e scrivere:
            docker exec -it lab_mariadb mariadb -u parser -pparserpass parserdb

            ---> SHOW TABLES;           per vedere le tabelle
            ---> SELECT COUNT(*) FROM web_resources;            per vedere se il numero di url corrisponde
            ---> SELECT url, domain FROM web_resources LIMIT 5;    per mostrare qualche url

            exit per uscire dalla modalità


---------------------------------------------------------------------
per restartare tutto:

    docker compose down
    sudo rm -rf ./mariadb_data/*                             sudo serve per i permessi
    docker compose up --build

