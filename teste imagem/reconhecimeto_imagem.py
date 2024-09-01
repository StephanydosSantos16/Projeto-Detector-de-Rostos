import cv2
import mysql.connector
import numpy as np
import time

class BancoDeDados:
    def __init__(self, host, user, password, database):
        self.conn = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.cursor = self.conn.cursor()

    def armazenar_foto(self, nome, foto):
        query = "INSERT INTO fotos_rostos (nome, foto) VALUES (%s, %s)"
        self.cursor.execute(query, (nome, foto))
        self.conn.commit()

    def obter_fotos(self):
        query = "SELECT nome, foto FROM fotos_rostos"
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def fechar(self):
        self.conn.close()

class DetetorDeRosto:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detectar_e_capturar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

class CapturadorDeFotos:
    def __init__(self, banco_de_dados):
        self.banco_de_dados = banco_de_dados
        self.detetor = DetetorDeRosto()
        self.cap = cv2.VideoCapture(0)
        self.classificador = cv2.face.LBPHFaceRecognizer_create()

        if not self.cap.isOpened():
            raise Exception("Não foi possível abrir a webcam. Verifique a conexão.")

    def capturar_foto(self, nome):
        print("Posicione seu rosto na frente da webcam.")

        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Não foi possível capturar a imagem. Tente novamente.")
                break

            faces = self.detetor.detectar_e_capturar(frame)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    cx, cy = x + w // 2, y + h // 2
                    frame_center = (frame.shape[1] // 2, frame.shape[0] // 2)
                    distance_to_center = np.sqrt((cx - frame_center[0]) ** 2 + (cy - frame_center[1]) ** 2)
                    
                    print(f'Distance to center: {distance_to_center}')
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    cv2.putText(frame, 'Rosto detectado. Posicione-se para a foto...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if distance_to_center < 150:
                        elapsed_time = time.time() - start_time
                        if elapsed_time > 5:
                            face_img = frame[y:y+h, x:x+w]
                            _, buffer = cv2.imencode('.jpg', face_img)
                            foto_bytes = buffer.tobytes()
                            self.banco_de_dados.armazenar_foto(nome, foto_bytes)
                            
                            cv2.putText(frame, 'Foto Capturada. Fechando...', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.imshow('Cadastro de Rosto', frame)
                            cv2.waitKey(2000)
                            
                            self.cap.release()
                            cv2.destroyAllWindows()
                            return

            cv2.imshow('Cadastro de Rosto', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def identificar_rosto(self):
        print("Posicione seu rosto na frente da webcam.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Não foi possível capturar a imagem. Tente novamente.")
                break

            faces = self.detetor.detectar_e_capturar(frame)
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    
                    # Carregar fotos do banco de dados
                    fotos_cadastradas = self.banco_de_dados.obter_fotos()
                    nomes = [foto[0] for foto in fotos_cadastradas]
                    fotos = [np.array(cv2.imdecode(np.frombuffer(foto[1], np.uint8), cv2.IMREAD_GRAYSCALE)) for foto in fotos_cadastradas]

                    if fotos:
                        # Treinar o classificador com as fotos cadastradas
                        ids = list(range(len(fotos)))
                        self.classificador.train(fotos, np.array(ids))
                        
                        # Identificar o rosto
                        label, confidence = self.classificador.predict(gray_face)
                        
                        # Ajustar o limite de confiança
                        if confidence < 100:  # Limite baixo para demonstração, ajuste conforme necessário
                            nome_identificado = nomes[label]
                            cv2.putText(frame, f'Identificado: {nome_identificado}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                        else:
                            cv2.putText(frame, 'Rosto não identificado', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
            cv2.imshow('Identificação de Rosto', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def iniciar(self):
        while True:
            opcao = input("Pressione 's' para cadastrar um rosto ou 't' para identificar um rosto (ou 'q' para sair): ").lower()
            if opcao == 's':
                nome = input("Digite o nome do usuário para cadastrar o rosto: ")
                self.capturar_foto(nome)
            elif opcao == 't':
                self.identificar_rosto()
            elif opcao == 'q':
                break
            else:
                print("Opção inválida. Tente novamente.")

def main():
    db = BancoDeDados(host='localhost', user='stephany', password='12345', database='gestures_db')
    capturador = CapturadorDeFotos(db)
    capturador.iniciar()
    db.fechar()

if __name__ == "__main__":
    main()
