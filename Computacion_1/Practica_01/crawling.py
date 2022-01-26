import requests
import re
parameters = {"principal page": "https://presidente.gob.mx/secciones/version-estenografica/page/",
              "regex": "(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?",
              "header": {"User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:96.0) Gecko/20100101 Firefox/96.0"},
              "path output": "Data/",
              "folder page": "pages/",
              "number of page": 92}
for page_i in range(1, parameters["number of page"]+1, 1):
    page = "{}{}/".format(parameters["principal page"],
                          page_i)
    page_i = str(page_i).zfill(2)
    filename_output = "{}{}{}.html".format(parameters["path output"],
                                           parameters["folder page"],
                                           page_i)
    responde = requests.get(page,
                            headers=parameters["header"])
    file = open(filename_output, "w")
    file.write(responde.text)
    file.close()
