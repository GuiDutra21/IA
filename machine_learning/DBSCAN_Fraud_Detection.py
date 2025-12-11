import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd

def generate_normal_transactions(n=800):
    """Gera transações normais (salário, compras, contas)"""
    np.random.seed(42)
    
    # Transações típicas durante horário comercial
    hours = np.random.normal(14, 4, n)  # Pico às 14h
    hours = np.clip(hours, 6, 23)
    
    # Valores típicos (R$ 2 a R$ 5000)
    amounts = np.random.lognormal(4, 1, n)  # Distribuição log-normal
    amounts = np.clip(amounts, 2, 5000)
    
    # Frequência (transações por dia: 1-5)
    frequency = np.random.poisson(3, n)
    frequency = np.clip(frequency, 1, 5)
    
    return np.column_stack((hours, amounts, frequency))

def generate_suspicious_patterns():
    """Gera padrões suspeitos de fraude"""
    suspicious = []
    
    # Padrão 1: Transações altas na madrugada
    n_laundering = 60
    hours_laundering = np.random.uniform(1, 5, n_laundering)  # 1h-5h da manhã
    amounts_laundering = np.random.uniform(2000, 8000, n_laundering)  # R$ 3k-15k
    freq_laundering = np.random.uniform(8, 15, n_laundering)  # Muitas transações
    suspicious.append(np.column_stack((hours_laundering, amounts_laundering, freq_laundering)))
    
    # Padrão 2: Múltiplas transações pequenas em sequência (teste de cartão)
    n_testing = 50
    hours_testing = np.random.uniform(2, 6, n_testing)
    amounts_testing = np.random.uniform(1, 5, n_testing)  # Valores muito baixos
    freq_testing = np.random.uniform(10, 20, n_testing)  # Frequência muito alta
    suspicious.append(np.column_stack((hours_testing, amounts_testing, freq_testing)))
    
    # Padrão 3: Transferências internacionais altas (atípico)
    n_international = 40
    hours_international = np.random.uniform(20, 24, n_international)  # Noite
    amounts_international = np.random.uniform(5000, 20000, n_international)
    freq_international = np.random.uniform(6, 10, n_international)
    suspicious.append(np.column_stack((hours_international, amounts_international, freq_international)))
    
    return np.vstack(suspicious)

def generate_random_outliers(n=80):
    """Gera outliers aleatórios (comportamentos completamente atípicos)"""
    hours = np.random.uniform(0, 24, n)
    amounts = np.random.uniform(0, 25000, n)
    frequency = np.random.uniform(0, 25, n)
    return np.column_stack((hours, amounts, frequency))

def main():
    
    # 1. Gerar transacoes simuladas
    normal = generate_normal_transactions(800)
    suspicious = generate_suspicious_patterns()
    outliers = generate_random_outliers(80)
    
    # Concatenar todos os dados
    X = np.vstack([normal, suspicious, outliers])
    
    # Labels reais (apenas para validação)
    labels_real = np.array(
        [0]*len(normal) +           # 0 = Normal
        [1]*len(suspicious) +        # 1 = Suspeito
        [2]*len(outliers)            # 2 = Outlier
    )
    
    print(f"Normal: {len(normal)} transações")
    print(f"Suspeitas: {len(suspicious)} transações")
    print(f"Outliers: {len(outliers)} transações")
    print(f"Total: {len(X)} transações")
    
    # 2. Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Aplicar DBSCAN
    dbscan = DBSCAN(eps=0.345, min_samples=15)
    clusters = dbscan.fit_predict(X_scaled)
    
    # Estatísticas
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)
    
    print(f"\nRESULTADOS:")
    print(f"Clusters (comportamentos padrão): {n_clusters}")
    print(f"Anomalias detectadas: {n_noise} ({n_noise/len(X)*100:.1f}%)")
    
    # Distribuição por cluster
    for i in range(n_clusters):
        count = list(clusters).count(i)
        print(f"   Cluster {i}: {count} transações")
    
    # Quantas suspeitas foram detectadas como anomalia?
    suspicious_indices = np.where(labels_real == 1)[0]
    suspicious_detected = np.sum(clusters[suspicious_indices] == -1)
    
    print(f"   Suspeitas identificadas: {suspicious_detected}/{len(suspicious_indices)} ({suspicious_detected/len(suspicious_indices)*100:.1f}%)")
    
    # Falsos positivos (normais marcadas como anomalia)
    normal_indices = np.where(labels_real == 0)[0]
    false_positives = np.sum(clusters[normal_indices] == -1)
    
    print(f"   Falsos positivos: {false_positives}/{len(normal_indices)} ({false_positives/len(normal_indices)*100:.1f}%)")
    
    # 5. Visualizações
    plt.figure(figsize=(18, 12))
    
    # Plot 1: Hora vs Valor (classificação REAL)
    ax1 = plt.subplot(2, 2, 1)
    colors_real = ['green', 'orange', 'red']
    labels_names = ['Normal', 'Suspeito', 'Outlier']
    for i in range(3):
        mask = labels_real == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=colors_real[i], 
                   label=labels_names[i], s=40, alpha=0.6, 
                   edgecolors='black', linewidth=0.5)
    ax1.set_xlabel('Hora do Dia (h)', fontsize=11)
    ax1.set_ylabel('Valor da Transação (R$)', fontsize=11)
    ax1.set_title('Classificação REAL\nHora × Valor', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 24)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Apenas anomalias detectadas
    ax2 = plt.subplot(2, 2, 2)
    anomalies_mask = clusters == -1
    
    ax2.scatter(X[anomalies_mask, 0], X[anomalies_mask, 1], 
               c='red', s=60, alpha=0.8, edgecolors='darkred', 
               linewidth=1.5, marker='X', label='Anomalia')
    ax2.set_xlabel('Hora do Dia (h)', fontsize=11)
    ax2.set_ylabel('Valor da Transação (R$)', fontsize=11)
    ax2.set_title(f'Anomalias Detectadas\n{n_noise} transações suspeitas', 
                 fontsize=12, fontweight='bold', color='red')
    ax2.set_xlim(0, 24)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distribuição de valores
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(X[anomalies_mask, 1], bins=30, alpha=0.8, color='red', 
            edgecolor='black', label='Anomalia')
    ax3.set_xlabel('Valor da Transação (R$)', fontsize=11)
    ax3.set_ylabel('Frequência', fontsize=11)
    ax3.set_title('Distribuição de Valores', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Análise textual
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Estatísticas de anomalias
    anomaly_data = X[anomalies_mask]
    if len(anomaly_data) > 0:
        avg_hour = np.mean(anomaly_data[:, 0])
        avg_amount = np.mean(anomaly_data[:, 1])
        avg_freq = np.mean(anomaly_data[:, 2])
    else:
        avg_hour = avg_amount = avg_freq = 0
    
    analysis = f"""
 RELATÓRIO DE FRAUDES

 ESTATÍSTICAS:
   • Total transações: {len(X)}
   • Anomalias: {n_noise} ({n_noise/len(X)*100:.1f}%)
   • Clusters normais: {n_clusters}

 PERFIL DAS ANOMALIAS:
   • Hora média: {avg_hour:.1f}h
   • Valor médio: R$ {avg_amount:.2f}
   • Freq. média: {avg_freq:.1f} trans/dia

 EFETIVIDADE:
   • Detecção: {suspicious_detected/len(suspicious_indices)*100:.1f}%
   • Falsos positivos: {false_positives/len(normal_indices)*100:.1f}%

  PADRÕES SUSPEITOS:
   1. Transações altas na madrugada
   2. Múltiplas transações pequenas
   3. Transferências atípicas
"""
    
    ax4.text(0.05, 0.95, analysis, fontsize=10, va='top', ha='left',
            bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9),
            family='monospace')
    
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
