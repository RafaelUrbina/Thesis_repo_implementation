import matplotlib.pyplot as plt
from matplotlib_venn import venn3

# Data breakdown
# Venn layout: (Abc, aBc, ABc, abC, AbC, aBC, ABC)
subsets = (1577, 2553, 532, 1803, 254, 406, 1106) 
labels = ('Technologies', 'Physical', 'Environmental')

# Exclusive totals mapped to their corresponding colors
exclusive_data = [
    {'label': 'Technologies', 'total': subsets[0], 'color': '#a8dadc'},
    {'label': 'Physical', 'total': subsets[1], 'color': '#457b9d'},
    {'label': 'Environmental', 'total': subsets[3], 'color': '#1d3557'}
]

# Sort the data by 'total' in ascending order so the largest bar appears at the top of a horizontal chart
exclusive_data_sorted = sorted(exclusive_data, key=lambda x: x['total'])

# Extract sorted values for the bar chart
sorted_labels = [item['label'] for item in exclusive_data_sorted]
sorted_totals = [item['total'] for item in exclusive_data_sorted]
sorted_colors = [item['color'] for item in exclusive_data_sorted]

# Monochromatic palette for the Venn diagram
monochrome_blue = ('#a8dadc', '#457b9d', '#1d3557') 

# Initialize a clean canvas
fig, ax = plt.subplots(figsize=(10, 8))

# ------------------------------------
# MAIN AREA: Venn Diagram
# ------------------------------------
v = venn3(subsets=subsets, set_labels=labels, set_colors=monochrome_blue, alpha=0.7, ax=ax)

# Format text inside the Venn diagram pockets
label_ids = ['100', '010', '110', '001', '101', '011', '111']
for label_id in label_ids:
    label = v.get_label_by_id(label_id)
    if label:
        label.set_fontsize(11)
        if label_id in ['110', '101', '011', '111']:
            label.set_weight('bold')
            label.set_color('#222222')

for text in v.set_labels:
    if text:
        text.set_fontsize(12)
        text.set_weight('bold')
        text.set_color('#1d3557')

# FIX 1: Attach the title directly to the main Venn diagram axes 'ax' so it centers perfectly over the circles
ax.set_title("Sub-topic(Dimensions) Distribution & Overlap Analysis", fontsize=16, fontweight='bold', pad=25, color='#1d3557')

# ------------------------------------
# UPPER RIGHT CORNER: Mini Proportional Bar Chart
# ------------------------------------
# Coordinates format: [left_distance, bottom_distance, width, height] 
ax_bar = fig.add_axes([0.75, 0.12, 0.12, 0.15])

# FIX 2: Plot using the sorted data and matching sorted colors
bars = ax_bar.barh(sorted_labels, sorted_totals, color=sorted_colors, alpha=0.8, height=0.55)

# Strip away clutter (borders and numbers) to keep it purely visual
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['bottom'].set_visible(False)
ax_bar.spines['left'].set_color('#cccccc') # Soft line next to category names

# Completely hide all numerical scales and data labels
ax_bar.get_xaxis().set_visible(False) 

# Format category labels to blend cleanly with the palette
ax_bar.tick_params(axis='y', labelsize=9.5, colors='#1d3557')
ax_bar.set_title('Exclusive Proportions', fontsize=10, fontweight='bold', color='#1d3557', pad=8)

plt.show()