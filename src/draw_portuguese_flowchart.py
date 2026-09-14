#!/usr/bin/env python3
"""
NBA Competitive Archetypes: Infographic Flowchart in Portuguese (TCC Quality)
Author: Data Visualization Specialist

This script draws a high-resolution (300 DPI) professional infographic containing
the 4 NBA team competitive archetypes, using rounded cards, custom hex colors,
and perfectly formatted Portuguese text.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
from pathlib import Path

def draw_flowchart(output_path: Path):
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(13, 10.5), facecolor='#f8f9fa')
    ax.set_facecolor('#f8f9fa')
    
    # Hide axes
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    # 1. Title and Subtitle
    plt.text(50, 95, "TAXONOMIA DE ESTILOS DE JOGO E COMPETITIVIDADE NA NBA (2019-2025)", 
             fontsize=17, fontweight='bold', color='#1a252f', ha='center', va='center')
    plt.text(50, 91, "Classificação e caracterização dos times em 4 perfis estratégicos fixos (Pooled K-Means)", 
             fontsize=11.5, fontstyle='italic', color='#5a6c7d', ha='center', va='center')
    
    # 2. Define Card contents in Portuguese
    cards_data = [
        {
            # TOP LEFT: CLUSTER 3
            'title': "Cluster 3: Elite",
            'bg_color': '#e8f5e9', # soft green
            'border_color': '#2e7d32',
            'text_color': '#1b5e20',
            'x': 3, 'y': 51, 'width': 45, 'height': 31,
            'profile': "Equipes com equilíbrio ideal entre ataque e defesa, eficiência altíssima, jogadores de elite e a capacidade de elevar o nível de jogo quando a temporada aperta.",
            'metrics': [
                ("Net Rating", "+3.43 (o maior da liga)"),
                ("Ajustado por Força (SRS)", "+1.16 (positivo contra os melhores)"),
                ("Eficiência Ofensiva", "115.74 de Off Rating (o melhor ataque)"),
                ("Média de Pontos", "116.40 pontos marcados por jogo"),
                ("Tendência Temporal", "+2.14 (crescem na reta final)"),
                ("Jogos Apertados (Clutch)", "Aproveitamento de 52.29%")
            ],
            'examples': "Boston Celtics (2021-25), Cleveland Cavaliers (2024-25)."
        },
        {
            # TOP RIGHT: CLUSTER 2
            'title': "Cluster 2: Offensive",
            'bg_color': '#fffde7', # soft yellow
            'border_color': '#f57f17',
            'text_color': '#e65100',
            'x': 52, 'y': 51, 'width': 45, 'height': 31,
            'profile': "Equipes que priorizam a transição rápida e o volume de arremessos de 3 pontos. Pontuam muito, mas a defesa vulnerável impede que cheguem ao nível de elite.",
            'metrics': [
                ("Net Rating", "+1.28 (positivo, menor que os defensivos)"),
                ("Ritmo de Jogo (Pace)", "102.17 (o mais alto da liga)"),
                ("Média de Pontos", "114.37 pontos marcados por jogo"),
                ("Consistência de Estilo", "13.00 de desvio padrão no Net Rating")
            ],
            'examples': "Atlanta Hawks (2018-21), New Orleans Pelicans (Zion jovem)."
        },
        {
            # BOTTOM LEFT: CLUSTER 1
            'title': "Cluster 1: Defensive",
            'bg_color': '#e1f5fe', # soft blue
            'border_color': '#0288d1',
            'text_color': '#01579b',
            'x': 3, 'y': 12, 'width': 45, 'height': 31,
            'profile': "Equipes competitivas que confiam em sua estrutura defensiva, desaceleram o adversário e controlam o erro. São consistentes e evoluem com o tempo.",
            'metrics': [
                ("Net Rating", "+1.57 (positivo)"),
                ("Eficiência Defensiva", "111.14 de Def Rating (a melhor da liga)"),
                ("Ritmo de Jogo (Pace)", "98.77 (o mais baixo entre os clusters)"),
                ("Média Pontos Sofridos", "109.70 (menor média permitida)"),
                ("Tendência Temporal", "+1.06 (evoluem na reta final da temporada)")
            ],
            'examples': "Boston Celtics (2018-19), Cleveland Cavaliers (2022-23), Chicago Bulls (2022-23)."
        },
        {
            # BOTTOM RIGHT: CLUSTER 0
            'title': "Cluster 0: Rebuilding",
            'bg_color': '#ffebee', # soft red
            'border_color': '#c62828',
            'text_color': '#b71c1c',
            'x': 52, 'y': 12, 'width': 45, 'height': 31,
            'profile': "Equipes em processo de reconstrução, com elencos jovens ou veteranos inativos, ineficientes nos dois lados da quadra e que desistem competitivamente no final do ano.",
            'metrics': [
                ("Net Rating", "-7.09 (péssimo)"),
                ("Eficiência Defensiva", "115.03 de Def Rating (pior da liga)"),
                ("Tendência Temporal", "-3.79 (pioram ao longo do ano — tanking)"),
                ("Jogos Apertados (Clutch)", "Aproveitamento de apenas 43.54%")
            ],
            'examples': "Golden State Warriors (2019-20, pós-lesões), Houston Rockets (2021-22)."
        }
    ]
    
    # 3. Draw cards
    for card in cards_data:
        # Draw rounded bounding box
        rect = patches.FancyBboxPatch(
            (card['x'], card['y']), card['width'], card['height'],
            boxstyle="round,pad=1.5",
            facecolor=card['bg_color'],
            edgecolor=card['border_color'],
            linewidth=1.8,
            zorder=1
        )
        ax.add_patch(rect)
        
        # Title of Card
        plt.text(card['x'] + 1, card['y'] + card['height'] - 1, card['title'],
                 fontsize=12, fontweight='bold', color=card['border_color'], ha='left', va='top', zorder=2)
        
        # Profile paragraph (wrapped)
        wrapped_profile = textwrap.fill(card['profile'], width=50)
        plt.text(card['x'] + 1, card['y'] + card['height'] - 4.5, wrapped_profile,
                 fontsize=9.5, color='#2c3e50', ha='left', va='top', linespacing=1.3, zorder=2)
        
        # Separator line
        plt.plot(
            [card['x'] + 1, card['x'] + card['width'] - 1],
            [card['y'] + card['height'] - 12.5, card['y'] + card['height'] - 12.5],
            color=card['border_color'], alpha=0.3, linewidth=1, zorder=2
        )
        
        # Metrics section header
        plt.text(card['x'] + 1, card['y'] + card['height'] - 13.5, "MÉTRICAS CHAVE:",
                 fontsize=9, fontweight='bold', color=card['text_color'], ha='left', va='top', zorder=2)
        
        # Write Metrics list
        y_offset = card['y'] + card['height'] - 15.8
        for label, val in card['metrics']:
            # Bullet point and Bold label
            plt.text(card['x'] + 1.5, y_offset, f"• {label}:", 
                     fontsize=9, fontweight='bold', color='#34495e', ha='left', va='top', zorder=2)
            # Metric value
            plt.text(card['x'] + 18, y_offset, val, 
                     fontsize=9, color='#2c3e50', ha='left', va='top', zorder=2)
            y_offset -= 1.8
            
        # Draw examples section dynamically relative to y_offset
        y_offset_examples = y_offset - 0.5
        plt.text(card['x'] + 1.5, y_offset_examples, "Exemplos:", 
                 fontsize=9, fontweight='bold', color='#34495e', ha='left', va='top', zorder=2)
        
        wrapped_examples = textwrap.fill(card['examples'], width=40)
        plt.text(card['x'] + 9.5, y_offset_examples, wrapped_examples, 
                 fontsize=9, fontstyle='italic', color='#2c3e50', ha='left', va='top', zorder=2)

    # 4. Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
    plt.close()
    print("Flowchart image rendered successfully in Portuguese.")

if __name__ == '__main__':
    project_dir = Path(__file__).resolve().parent.parent
    viz_folder = project_dir / "viz"
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    output_file = viz_folder / "nba_clusters_flowchart_pt.png"
    draw_flowchart(output_file)
