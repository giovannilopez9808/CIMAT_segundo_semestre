import requests
url_principal_page = "https://presidente.gob.mx/secciones/version-estenografica/"
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0"}
responde = requests.get(url_principal_page, headers=headers)
file = open("Data/conferencia.txt", "w")
file.write(responde.text)
file.close()
