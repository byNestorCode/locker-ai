import cv2
import os

# Para reconocer rostros de un bando de imagenes:
# Obtener el directorio base
BASEDIR = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists("banco/"):
        print("Creando carpeta de Banco de imágenes...")
        os.makedirs("banco/") 
        print("Carpeta creada con éxito!")

imagesPath = os.path.join(BASEDIR, 'banco')
imagesPathList = os.listdir(imagesPath)
print('imagesPathList=', imagesPathList)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

count = 0

for imageName in imagesPathList:
        if not imageName:
            print("Tienes que ingresar imágenes al directorio...")
        else:
            print('imageName=', imageName)
            image = cv2.imread(imagesPath+'/'+imageName)
            """ cv2.imshow('image', image)
            cv2.waitKey(0) """
            imageAux = image.copy()
            gray_scale =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceClassif.detectMultiScale(gray_scale, 1.1, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(image,(x,y),(x+w,y+h),(128,0,255),2)
                cv2.rectangle(image,(10,5),(500,25),(255,255,355),-1)
                cv2.putText(image, 'Presione la tecla s para almacenar los rostros encontrados',(10,20),2,0.5,(128,0,255),1,cv2.LINE_AA)
                cv2.imshow('image', image)

                k = cv2.waitKey(0)
                if k == ord('s'):
                    if not os.path.exists('rostros/'):
                        print("Creando carpeta de rostros...")
                        os.makedirs("rostros/") 
                        print("Carpeta creada con éxito!")


                    for (x,y,w,h) in faces:
                        rostro = imageAux[y:y+h,x:x+w]
                        rostro = cv2.resize(rostro,(150, 150), interpolation=cv2.INTER_CUBIC)
                        # Pruebas de visualizacion de rostros detectados
                        #cv2.imshow('rostro', rostro)
                        #cv2.waitKey(0)
                        cv2.imwrite(BASEDIR+'/rostros/'+'rostro_{}.jpg'.format(count),rostro)
                        count = count + 1
                        print('Rostro guardado')
                elif k == 27:
                    break

print("Finalizó la ejecución")
cv2.destroyAllWindows()