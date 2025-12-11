"""
Interface Gráfica Simples para Reconhecimento de Dígitos (MNIST)
Carregue uma imagem e classifique com CNN
"""

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
from tensorflow.keras.models import load_model
import os

class MNISTImageClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Classificador de Dígitos")
        self.root.geometry("450x700")
        self.root.configure(bg='white')
        
        # Carregar modelo
        self.model = self.load_model()
        self.processed_image = None
        
        # Criar interface
        self.create_widgets()
        
    def load_model(self):
        """Carrega o modelo treinado"""
        model_path = 'mnist_cnn_model.h5'
        if os.path.exists(model_path):
            try:
                model = load_model(model_path)
                print(f"✓ Modelo carregado")
                return model
            except Exception as e:
                messagebox.showerror("Erro", f"Erro ao carregar modelo:\n{e}")
                return None
        else:
            messagebox.showwarning("Aviso", "Modelo não encontrado!\nExecute 'cnn_digitos.py' primeiro.")
            return None
    
    def create_widgets(self):
        """Cria a interface"""
        
        # Título
        tk.Label(
            self.root,
            text="Classificador de Dígitos",
            font=('Arial', 18, 'bold'),
            bg='white'
        ).pack(pady=15)
        
        # Botão Carregar
        tk.Button(
            self.root,
            text="Carregar Imagem",
            font=('Arial', 12),
            bg='#3498DB',
            fg='white',
            command=self.load_and_predict,
            width=20,
            height=2
        ).pack(pady=10)
        
        # Canvas para imagem
        self.canvas = tk.Canvas(
            self.root,
            width=280,
            height=280,
            bg='#F0F0F0',
            highlightthickness=1,
            highlightbackground='#CCC'
        )
        self.canvas.pack(pady=10)
        
        # Resultado
        self.result_label = tk.Label(
            self.root,
            text="Nenhuma imagem carregada",
            font=('Arial', 20, 'bold'),
            bg='white',
            fg='#333'
        )
        self.result_label.pack(pady=20)
        
        # Probabilidades (texto simples)
        self.prob_text = tk.Label(
            self.root,
            text="",
            font=('Courier', 9),
            bg='white',
            fg='#666',
            justify=tk.LEFT
        )
        self.prob_text.pack(pady=10)
    
    def load_and_predict(self):
        """Carrega imagem e faz predição automaticamente"""
        if self.model is None:
            messagebox.showerror("Erro", "Modelo não carregado!")
            return
        
        # Abrir diálogo
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp"), ("Todos", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            # Carregar e processar imagem
            img = Image.open(file_path).convert('L')  # Grayscale
            
            # Exibir imagem
            display_img = img.copy()
            display_img.thumbnail((280, 280), Image.LANCZOS)
            photo = ImageTk.PhotoImage(display_img)
            self.canvas.delete('all')
            self.canvas.create_image(140, 140, image=photo)
            self.canvas.image = photo
            
            # Processar para modelo
            img_28 = img.resize((28, 28), Image.LANCZOS)
            
            # Inverter cores se fundo claro
            if np.mean(np.array(img_28)) > 127:
                img_28 = ImageOps.invert(img_28)
            
            self.processed_image = img_28
            
            # Fazer predição
            self.predict()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro: {e}")
    
    def predict(self):
        """Faz a predição"""
        if self.processed_image is None:
            return
        
        try:
            # Preparar imagem
            img_array = np.array(self.processed_image).astype('float32') / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)
            
            # Predição
            predictions = self.model.predict(img_array, verbose=0)[0]
            predicted_digit = np.argmax(predictions)
            confidence = predictions[predicted_digit] * 100
            
            # Atualizar resultado
            self.result_label.config(
                text=f"Dígito: {predicted_digit} ({confidence:.1f}%)",
                fg='green' if confidence > 70 else 'orange'
            )
            
            # Mostrar probabilidades
            prob_lines = []
            for i in range(10):
                bar = '█' * int(predictions[i] * 15)
                prob_lines.append(f"{i}: {predictions[i]*100:5.1f}% {bar}")
            self.prob_text.config(text='\n'.join(prob_lines))
            
            # Console
            print(f"\n✓ Predição: {predicted_digit} ({confidence:.1f}%)")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na predição: {e}")

def main():
    """Função principal"""
    root = tk.Tk()
    app = MNISTImageClassifier(root)
    root.mainloop()

if __name__ == "__main__":
    main()
