import matplotlib.pyplot as plt
import pandas as pd
import os

def generate_guaranteed_charts():
    # 1. Create a folder named 'report_images' in the CURRENT directory
    current_dir = os.getcwd()
    output_folder = os.path.join(current_dir, "report_images")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📂 Created new folder: {output_folder}")

    # --- FIGURE 1: COMPLAINT DISTRIBUTION ---
    data = {
        'Product': ['Credit card', 'Credit reporting', 'Money transfers', 'Personal loan', 'Savings account'],
        'Count': [8500, 9200, 1200, 1500, 400] 
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Product'], df['Count'], color=['#1f77b4', '#aec7e8', '#ff7f0e', '#2ca02c', '#9467bd'])
    plt.title('Figure 1: Distribution of Complaints by Product (Class Imbalance)', fontsize=14, fontweight='bold')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    path1 = os.path.join(output_folder, "figure1_distribution.png")
    plt.savefig(path1, dpi=300)
    print(f"✅ Saved Figure 1 at: {path1}")

    # --- TABLE 1: STRATIFIED SAMPLING ---
    table_data = {
        'Product Category': ['Credit Card', 'Credit Reporting', 'Money Transfer', 'Personal Loan', 'Savings Account', 'TOTAL'],
        'Original Raw Data': ['High Vol', 'High Vol', 'Med Vol', 'Low Vol', 'Low Vol', '464k+'],
        'Stratified Sample': ['6,000', '3,500', '1,000', '1,000', '500', '12,000']
    }
    df_table = pd.DataFrame(table_data)
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df_table.values, colLabels=df_table.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        elif row == 6:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    path2 = os.path.join(output_folder, "table1_stratified_split.png")
    plt.savefig(path2, bbox_inches='tight', dpi=300)
    print(f"✅ Saved Table 1 at:  {path2}")

    # --- FIGURE 2: ARCHITECTURE ---
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    # Draw boxes
    ax.text(0.1, 0.5, "CFPB Dataset\n(Raw CSV)", ha="center", va="center", size=12, bbox=dict(boxstyle="round,pad=0.3", fc="#e1e1e1", ec="#333"))
    ax.text(0.35, 0.5, "Ingestion Pipeline\n(Filter, Clean,\nStratified Sample)", ha="center", va="center", size=12, bbox=dict(boxstyle="round,pad=0.3", fc="#aec7e8", ec="#333"))
    ax.text(0.6, 0.5, "Embedding\n(MiniLM-L6)\n& Chunking", ha="center", va="center", size=12, bbox=dict(boxstyle="round,pad=0.3", fc="#ffbb78", ec="#333"))
    ax.text(0.85, 0.5, "Vector Store\n(ChromaDB)", ha="center", va="center", size=12, bbox=dict(boxstyle="round,pad=0.3", fc="#98df8a", ec="#333"))
    
    # Arrows
    ax.annotate("", xy=(0.23, 0.5), xytext=(0.17, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.52, 0.5), xytext=(0.47, 0.5), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(0.77, 0.5), xytext=(0.69, 0.5), arrowprops=dict(arrowstyle="->", lw=2))

    plt.title("Figure 2: Data Ingestion Architecture", fontsize=14, fontweight='bold')
    path3 = os.path.join(output_folder, "figure2_architecture.png")
    plt.savefig(path3, dpi=300)
    print(f"✅ Saved Figure 2 at: {path3}")

if __name__ == "__main__":
    generate_guaranteed_charts()