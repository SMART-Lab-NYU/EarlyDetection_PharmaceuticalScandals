import re
import sys
import csv
import threading
import timeit
import time
import pandas as pd
import urllib.request
import unicodedata
import re
import pandas as pd
from google.colab import files

class GetAllPages_topic_Thread(threading.Thread):

    def __init__(self, turl, page, nb_page, sujet):
        threading.Thread.__init__(self)
        self.url = turl
        self.page = page
        self.nb_page = nb_page
        self.sujet = sujet

    def run(self):
        global all_msg
        if debug:
          print("[Ouverture URL de \"" + self.sujet + "\" page " + str(page) + "] : " + str(self.url))
        
        html = get_html(url, 5)
        if html == -1:
          print(f"[Error get messages] --> abort page {url}")
          return
       
        only_messages = re.search('<div id="topic" >(.*?)<div class="bottom_action_topic_menu">', html, re.MULTILINE | re.DOTALL).group(1)
        messages_page = re.findall('class="md-topic_post(.*?)/table>', only_messages, re.MULTILINE | re.DOTALL)

        for message in messages_page:
          if re.match('.*data-id_user.*', message, re.DOTALL):
            user = re.search('data-id_user.+?>(.+?)<', message).group(1)
          elif re.match('.*itemprop="name"', message, re.DOTALL):
              user = re.search('itemprop="name".*?>(.+?)<', message).group(1)
          elif re.match('.*Profil supprimé.*', message, re.DOTALL):
              user = "Profil supprimé"
          else:
              user = "[ERROR_Encodage_user_unknown]"
          date = re.search('Posté le ([0-9/]+)', message).group(1)
          if re.match('.*itemprop="citation".*', message, re.DOTALL):
            message = re.sub('itemprop="citation".+?</span></span>', '', message, flags=re.DOTALL)
          text = re.search('itemprop="text" hidden>(.*?)</span><div>', message, re.MULTILINE | re.DOTALL).group(1)
          text = clean_message(text)
          all_msg = all_msg.append({'date':date, 'user':user, 'text':text, 'url':self.url}, ignore_index=True)

        if debug:
          print("[" + str(len(messages_page)) + " new msg sur \"" + self.sujet + "\" de la page " + str(self.page) + " sur " + self.nb_page + "]")


def clean_message(msg):
    msg = re.sub('&#039;', '\'', msg)#apostrop"he
    msg = re.sub(',', ' ', msg)  # Pour un decoupage correct sur excel
    msg = re.sub('[>\r\n]+', ' ', msg) #Saut de ligne
    msg = re.sub(':\w+:', ' ', msg) #les smiley :happy:
    msg = re.sub('http\://.+?\.html', '', msg) #les liens copi
    msg = re.sub('<img.*?/>', ' ', msg)  #suppr les images
    msg = re.sub('<br.*?>', ' ', msg) #suppr les balise br
    msg = re.sub('<a (.*?)</a>', ' ', msg) #suppr les liens externe
    msg = re.sub('</?span.*?>', ' ', msg)
    msg = re.sub('</?table.*?>', ' ', msg)
    msg = re.sub('</?[a-z][a-z]?>', ' ', msg) #</i> <lu> et bien d'autre
    msg = re.sub('&[a-z#0-9]{1,4};', ' ', msg) #&#034; &nbsp; &euro; &gt; &lt;
    msg = re.sub('\[#[0-9]+ size=[0-9]+\]', ' ', msg)
    msg = re.sub('</?strong>', ' ', msg)
    msg = re.sub('</?div>?', ' ', msg)
    msg = unicodedata.normalize('NFD', msg).encode('ascii', 'ignore')  # suppr les accents
    return(msg)

    def get_nbr_page(html):
    list_pages = re.search('pagination_main_visible(.+?)/div', html).group(1)
    if re.match(r".*href.*", list_pages):
        return(re.findall("\">([0-9]+)<", html)[-1])
    else:
        return("1")
        
def get_html(url:str, max_attempt:int):
  attempt = 1
  while (attempt <= max_attempt):
    try:
      with urllib.request.urlopen(url) as response:
        html = response.read().decode('utf-8')
        return html
    except OSError as e:
      print(f"[Error {e.code}] {e.reason} : {url}")
      if e.code == 503:
        time.sleep(60)
      time.sleep(1)
      attempt += 1
  return -1

# Executiuon start here
search = "levothyrox"
rubrique = "18*sante"
debug = True
save_tmp_out = True

print("Recherche de <"+search+"> dans la rubrique <"+rubrique+">")
if debug:
    print('http://forum.doctissimo.fr/search_result.php?post_cat_list='+rubrique+'&search='+search+'&resSearch=250')
with urllib.request.urlopen('http://forum.doctissimo.fr/search_result.php?post_cat_list='+rubrique+'&search='+search+'&resSearch=250') as response:
  html = response.read().decode('utf-8')
if re.match(r".*aucune réponse n'a été trouvée.*", html, re.MULTILINE|re.DOTALL):
    print("La recherche de <"+search+"> dans la rubrique <"+rubrique+"> donne aucun résultat")
    sys.exit()
nb_page_topic = get_nbr_page(html)

print(nb_page_topic+" page(s) de 250 topics sur le sujet <"+search+"> dans la rubrique <"+rubrique+">")

all_topics_url = []
page = 1
if debug:
    nb_page_topic = "1"
while page <= int(nb_page_topic):
    print("telechargement de page " + str(page) + " sur " + nb_page_topic)
    if debug:
        print('http://forum.doctissimo.fr/search_result.php?post_cat_list='+rubrique+'&search='+search+'&resSearch=250&page='+str(page))
    with urllib.request.urlopen('http://forum.doctissimo.fr/search_result.php?post_cat_list='+rubrique+'&search='+search+'&resSearch=250&page='+str(page)) as response:
        html = response.read().decode('utf-8')
    topics = re.findall(r"</?t.*?sujet ligne_booleen(.+?)</tr>", html, re.MULTILINE | re.DOTALL)
    for topic in topics:
        if debug:
            print(re.search(r"href=\"(.+?)\"", topic).group(1))
        all_topics_url.append(re.search(r"href=\"(.+?)\"", topic).group(1))
    page += 1
print("nb total de topic = " + str(len(all_topics_url)))

start = timeit.default_timer()
all_msg = pd.DataFrame(columns=['date','user', 'text', 'url'])
threadList = []
nb_topic = 0

for url in all_topics_url:  
    if debug:
        print(url)
        time.sleep(2)
    html = get_html(url, 5)

    if html == -1:
      print(f"[Error get nb page] --> abort topic {url}")
      continue 

    sujet_topic = re.search("forum.doctissimo.fr/sante/.+/(.*?)sujet_", url).group(1)
    nb_page_topic = get_nbr_page(html)

    if debug:
        print("topic \""+sujet_topic+"\" avec "+str(nb_page_topic)+" page(s)")
    page = 1
    while page <= int(nb_page_topic):
        clean_url = re.search(r"(.*)_", url).group(1)
        newthread = GetAllPages_topic_Thread(clean_url + "_" + str(page) + ".htm", page, nb_page_topic, sujet_topic)
        newthread.start()
        time.sleep(0.1)
        threadList.append(newthread)
        page += 1

    if (nb_topic % 100 == 0):
      print(str(nb_topic) + " topics extrait sur " + str(len(all_topics_url)) + ". Messages récoltés : " + str(len(all_msg)))
    
    if len(all_msg) > 1000 and save_tmp_out == True:
        all_msg.to_csv("tmp_out.csv", sep=',', encoding='utf-8', index=False)
        print("Fichier temporaire save --> tmp_out.csv")
        save_tmp_out = False
    nb_topic += 1

print("Attente des threads")
for curThread in threadList :
    curThread.join()
all_msg.to_csv("out.csv", sep=',', encoding='utf-8', index=False)
stop = timeit.default_timer()
m, s = divmod(stop - start, 60)
h, m = divmod(m, 60)
print(str(len(all_msg)) + " messages total récoltés en %dh %02dmin et %02ds" % (h, m, s))

url = "http://forum.doctissimo.fr/sante/arthrose-os/maigrir-sujet_149370_1.htm" # profil supprimé + citation normol
url = "http://forum.doctissimo.fr/sante/thyroide-problemes-endocrinologiques/endocrinologue-belgique-sujet_160644_1.htm" # encodage user avec space authorisé (Susanne in F)
url = "http://forum.doctissimo.fr/sante/thyroide-problemes-endocrinologiques/supportez-thyroxine-sanofi-sujet_171008_1.htm" #avec hidden dans code devant user name
url = "http://forum.doctissimo.fr/sante/thyroide-problemes-endocrinologiques/demande-renseignements-sujet_171692_1.htm" #encodage citaiton different
url = "http://forum.doctissimo.fr/sante/regles-problemes-gynecologiques/retart-regles-sujet_222033_1.htm" # encodage citation différent
url = "http://forum.doctissimo.fr/sante/thyroide-problemes-endocrinologiques/probleme-couple-tyroide-sujet_152716_2.htm" #message de fay41ft le 24/04/2006 "..." cité absent sur la page web mais présent dans le tableau ??
url = "https://forum.doctissimo.fr/sante/thyroide-problemes-endocrinologiques/interruption-pituitaire-hypophysaire-sujet_156698_1.htm"

df = pd.DataFrame(columns=['date','user', 'text', 'url'])
try:
    with urllib.request.urlopen(url) as rep:
        html = rep.read().decode('utf-8')
except (http.client.IncompleteRead) as e:
    html = e.partial.decode('utf-8')

only_messages = re.search('<div id="topic" >(.*?)<div class="bottom_action_topic_menu">', html, re.MULTILINE | re.DOTALL).group(1)
messages_page = re.findall('class="md-topic_post(.*?)/table>', only_messages, re.MULTILINE | re.DOTALL)
for message in messages_page:
    if re.match('.*data-id_user.*', message, re.DOTALL):
        user = re.search('data-id_user.+?>(.+?)<', message).group(1)
    elif re.match('.*itemprop="name"', message, re.DOTALL):
        user = re.search('itemprop="name".*?>(.+?)<', message).group(1) #parfois hidden est rajouté dans le code source donc .+? après name
    elif re.match('.*Profil supprimé.*', message, re.DOTALL):
        user = "Profil supprimé"
    else:
        user = "[ERROR_Encodage_user_unknown]"
    date = re.search('Posté le ([0-9/]+)', message).group(1)
    if re.match('.*itemprop="citation".*', message, re.DOTALL):
        message = re.sub('itemprop\=\"citation\".+?</span><span', '', message, flags=re.DOTALL)
    text = re.search('itemprop="text" hidden>(.*?)</span>[<div>|<span itemprop="author"]', message, re.MULTILINE | re.DOTALL).group(1)
    text = clean_message(text)
    df = df.append({'date':date, 'user':user, 'text':text, 'url':url}, ignore_index=True)
print(str(len(messages_page)))