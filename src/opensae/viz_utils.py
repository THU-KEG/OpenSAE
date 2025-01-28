import json
import numpy as np
import os
from matplotlib import cm

def colorize_token(token, activation_value, max_activation_value):
    """
    Colorize a token based on its activation value using CSS for HTML rendering.
    """
    if max_activation_value == 0:
        normalized_activation = 0
    else:
        # Normalize the activation value to a range between 0 and 1
        normalized_activation = np.clip(activation_value / max_activation_value, 0, 1)

    scaled_activation = normalized_activation ** 2  

    red = int(255 * scaled_activation)
    green = int(255 * (1 - scaled_activation))
    blue = 0 

    # Construct the color style in RGB for HTML
    color_style = f"rgb({red}, {green}, {blue})"
    
    return f'<span style="color:{color_style};">{token}</span>'

def generate_html(activations, base_vector_indices):
    """
    Generate the HTML content for visualizing activation data with colorized sentences and bar charts.
    """
    html_content = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {
                font-family: "Microsoft YaHei", "Arial", sans-serif;
                background-color: #f4f4f4;
                display: flex;
                justify-content: center;
                padding: 20px;
            }
            .container {
                display: flex;
                width: 80%;
            }
            .sidebar {
                width: 100%;
                margin-right: 20px;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                position: sticky;
                top: 20px;
                height: 80vh;
                overflow-y: auto;
            }
            .vector-list {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .vector-item {
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                cursor: pointer;
                text-align: center;
                flex-shrink: 0;
            }
            .vector-item:hover {
                background-color: #ddd;
            }
            .content {
                width: 75%;
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                max-height: 80vh;
                overflow-y: auto;
            }
            .sentence {
                margin-bottom: 20px;
                font-size: 16px;
            }
            .sentence h4 {
                margin-bottom: 10px;
            }
            h2 {
                color: #333;
            }
            strong {
                color: #333;
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
        <script>
            function showVectorContent(baseVectorIndex) {
                var contentSections = document.querySelectorAll('.vector-content');
                contentSections.forEach(function(section) {
                    section.style.display = 'none';
                });
                var selectedContent = document.getElementById('vector-' + baseVectorIndex);
                selectedContent.style.display = 'block';
            }
        </script>
    </head>
    <body>
    <div class="container">
        <div class="sidebar">
            <h3>Base Vectors</h3>
            <div class="vector-list" id="vector-list">
    """

    # Add base vector indices to the sidebar
    for base_vector_index in base_vector_indices:
        html_content += f'<div class="vector-item" onclick="showVectorContent({base_vector_index})">Base Vector {base_vector_index}</div>'

    html_content += """
            </div>
        </div>

        <div class="content">
            <h2>Activation Visualization</h2>
    """

    # Generate content for each base vector
    for base_vector_index in base_vector_indices:
        html_content += f'<div class="vector-content" id="vector-{base_vector_index}" style="display:none;">'
        html_content += f"<h3>Visualizing base vector index {base_vector_index}</h3>"
        
        sentence_counter = 0
        for sentence_data in activations:
            sentence_counter += 1
            sentence_id = sentence_data.get("sentence_id", "Unknown")
            tokens = sentence_data.get("tokens", [])
            
            colorized_sentence = f'<div class="sentence"><strong>Sentence ID: {sentence_id}</strong><br>'

            token_list = []
            activation_list = []

            for token_data in tokens:
                token = token_data.get("token", "")
                token_list.append(token)
                activation = 0
                for top in token_data.get("activations", []):
                    if top.get("base_vector") == base_vector_index:
                        activation = top.get("activation", 0)
                        break
                activation_list.append(activation)
                colorized_token = colorize_token(token, activation, max(activation, 1))
                colorized_sentence += colorized_token + " "

            colorized_sentence += '</div>'
            html_content += colorized_sentence

            canvas_id = f"chart-{base_vector_index}-{sentence_counter}"
            html_content += f'<canvas id="{canvas_id}" width="400" height="200"></canvas>'

            tokens_js = json.dumps(token_list, ensure_ascii=False)
            activation_js = json.dumps(activation_list)

            html_content += f"""
            <script>
            (function() {{
                var ctx = document.getElementById("{canvas_id}").getContext("2d");
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {tokens_js},
                        datasets: [{{
                            label: 'Activation Value',
                            data: {activation_js},
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        plugins: {{
                            datalabels: {{
                                anchor: 'end',
                                align: 'end',
                                formatter: function(value) {{
                                    return value === 0 ? '' : value.toFixed(3);
                                }},
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }},
                    plugins: [ChartDataLabels]
                }});
            }})();
            </script>
            """

        html_content += "</div>"

    html_content += """
        </div>
    </div>
    </body>
    </html>
    """
    return html_content

def save_html_file(html_content, file_path):
    """
    Save the generated HTML content to a specified file.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
