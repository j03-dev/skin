from oxapy import HttpServer
import v1


def main():
    HttpServer(("0.0.0.0", 8000)).attach(v1.router()).run()


if __name__ == "__main__":
    main()
