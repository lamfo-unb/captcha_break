import ssl
import os
from urllib.request import urlretrieve
from urllib.request import urlopen
from bs4 import BeautifulSoup
import time
import deathbycaptcha
import certifi


# def cleanImage(imagePath):
#    image = Image.open(imagePath)
#    image = image.point(lambda x: 0 if x < 100 else 255)
#    borderImage = ImageOps.expand(image, border=20, fill='white')
#    borderImage.save(imagePath)

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

ativo = "ABEV3"
for i in range(0, 10000):
    html = urlopen('https://www.fundamentus.com.br/balancos.php?papel=' + ativo + '&tipo=1', cafile=certifi.where())

    bs = BeautifulSoup(html, 'html.parser')

    # Gather prepopulated form values
    imageLocation = bs.find('img', {'class': 'captcha'})['src']

    captchaUrl = 'https://www.fundamentus.com.br/' + imageLocation
    urlretrieve(captchaUrl, 'DeathByCaptcha/captcha.png')
    print(i)
    time.sleep(1)
    # cleanImage('captcha.png')


# Send the captcha to the DeathByCaptcha
    username = "pedrobsb"
    password = "******"

# print(len(os.listdir("DeathByCaptcha")))

#for filename in enumerate(os.listdir("DeathByCaptcha")):
    dst = 'captcha.png'
    #dst = "captcha" + str(count) + ".png"
    # Get the solution
    files = len(os.listdir('SolvedDBC'))
    client = deathbycaptcha.SocketClient(username, password)
    captcha = client.decode('DeathByCaptcha/' + dst, 15)
    if captcha["text"] is not None:
        solution = captcha["text"]
        time.sleep(1)
        os.rename('DeathByCaptcha/' + dst, 'SolvedDBC/' + solution + '.png')
        print("Changing", i, "to", solution)
    else:
        print('Change failed, trying another loop...:', i)
        continue

