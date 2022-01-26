from bs4 import BeautifulSoup
from pprint import pprint
import requests
import re
import os


def download(filename: str, page: str, header: dict):
    """
    Descarga de las paginas web usando request y headers 
    """
    responde = requests.get(page,
                            headers=header)
    file = open(filename, "w")
    file.write(responde.text)
    file.close()


def format_date(date: str) -> str:
    """
    formato de la fecha de dd.mm.yy a yyyy-mm-dd 
    """
    day, month, year = date.split(".")
    out = "20{}-{}-{}".format(year, month, day)
    return out


def format_filename(path: str, date: str) -> str:
    """
    Formato del nombre del archivo con su fecha y el numero de sesion del día 
    """
    files = os.listdir(path)
    part = sum([True for file in files if date.replace(".", "-") in file])
    if part:
        part = str(part+1)
    else:
        part = "1"
    date = format_date(date)
    date = "{}_{}".format(date, part)
    return date


parameters = {"page": "https://presidente.gob.mx",
              "principal page": "https://presidente.gob.mx/secciones/version-estenografica/page/",
              "regex url": "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
              "regex date": "[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]",
              "header": {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0"},
              "path output": "Data/",
              "folder page": "pages/",
              "folder corpus": "estenograficas/",
              "folder corpus clean": "clean_estenograficas/",
              "folder dataset": "dataset/",
              "number of page": 92}
urls = []
for page_i in range(1, parameters["number of page"]+1, 1):
    page = "{}{}/".format(parameters["principal page"],
                          page_i)
    page_i = str(page_i).zfill(2)
    filename_output = "{}{}{}.html".format(parameters["path output"],
                                           parameters["folder page"],
                                           page_i)
    # Descarga del indice de las sesiones
    download(filename_output,
             page,
             parameters["header"])
    file = open(filename_output,
                "r",
                encoding="utf-8")
    text = file.read()
    file.close()
    # Obtiene las urls de las sesiones
    urls += [path for protocol,
             domain,
             path in re.findall(parameters["regex url"],
                                text) if "estenografica-de-la" in path]

urls = sorted(list(set(urls)))
for url in urls:
    complete_url = "{}{}".format(parameters["page"],
                                 url)
    url = url.replace("/", "_")
    output = "{}{}{}.html".format(parameters["path output"],
                                  parameters["folder corpus"],
                                  url)
    # Descarga de cada sesion
    download(output,
             complete_url,
             parameters["header"])
    file = open(output,
                "r",
                encoding="utf-8")
    text = file.read()
    file.close()
    # Clean HTML
    soup = BeautifulSoup(text, "html.parser")
    output = "{}{}{}.dat".format(parameters["path output"],
                                 parameters["folder corpus clean"],
                                 url)
    file = open(output,
                "w",
                encoding="utf-8")
    text = file.write(soup.get_text())
    file.close()
    # Change name
    path_output = "{}{}".format(parameters["path output"],
                                parameters["folder dataset"])
    file = open(output,
                "r",
                encoding="utf-8")
    text = file.read()
    file.close()
    date = re.findall(parameters["regex date"],
                      text)
    # Guardado de archivos con la fecha en su nombre
    if len(date):
        date = date[0]
        filename = format_filename(path_output, date)
        output = "{}{}.dat".format(path_output,
                                   filename)
        file = open(output,
                    "w",
                    encoding="utf-8")
        file.write(text)
        file.close()
    else:
        print("Checar ", output)
