# captcha_break


Uso do modelo treinado para identificação de Captchas alfabeticos. 

O modelo foi treinado para Capthas de 5 letras cujo a imagem possui dimensões 60x200x1, conforme o exeplo abaixo.

![neil](captcha.png)

Arquivos: 
+ **Best_CaptchaModel.h5**: Modelo treinado.
+ **LabelBinarizer.joblib**: Labels das classificações
+ **requirements.txt**: Lista de pacotes utilizados

---
Exemplo de uso: 

```
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import cv2

lb = joblib.load("LabelBinarizer.joblib")
model = load_model('Best_CaptchaModel.h5')

img = cv2.imread('captcha.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # Converte para Escalas de cinza
img = (img - img.mean()) / img.std()            # Normalizacao da Imagem
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
cls_pred = lb.inverse_transform(np.array(pred).max(1))

print(cls_pred)
# ['W' 'P' 'X' 'W' 'T']
```
