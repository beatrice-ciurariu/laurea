# parser

comandi per far partire il backend da windows
> cd backend/src

> uvicorn server:app --reload --port 8003

comandi per far partire il fronend
> cd frontend/src

> uvicorn frontend:app --reload --port 8004

pagina web
http://localhost:8004/





LINUX--------------

    docker compose up --build

e attendere (un bel po') che scarichi tutto: si creeranno le cartelle mariadb_data e ollama_data


---------------------------------------------------------------------
per restartare tutto:

    docker compose down
    sudo rm -rf ./mariadb_data/*                             sudo serve per i permessi
    docker compose up --build

