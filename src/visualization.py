import math
import numpy as np
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from IPython.display import Video, HTML
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from jinja2 import Template
from .utils import get_random_files


def create_video_grid(
    video_paths, 
    video_annos=None,
    n_cols=2, 
    width=320
):
    n_videos = len(video_paths)
    n_rows = math.ceil(n_videos / n_cols)
    
    grid_html = f'''
    <div style="display: grid; 
                grid-template-columns: repeat({n_cols}, 1fr); 
                grid-template-rows: repeat({n_rows}, auto);
                gap: 10px;">
    '''
    
    for path in video_paths:
        video = Video(path, width=width, embed=True)
        grid_html += f'<div>{video._repr_html_()}'
        if video_annos is not None:
            caption_str = ''
            for caption_i in video_annos[path.name]:
                caption_str += f' - {caption_i}<br/>'
            grid_html += f'<p max-width: {width}px;">{caption_str}</p>'
        grid_html += '</div>'

    for _ in range(n_cols * n_rows - n_videos):
        grid_html += '<div></div>'
    
    grid_html += '</div>'
    return HTML(grid_html)


def visualize_random_files(
    video_paths,
    video_annos=None,
    n_samples=9,
    n_cols=3,
    width=320
):
    random_files = get_random_files(
        video_paths, 
        n=n_samples
    )
    return create_video_grid(
        random_files, 
        video_annos=video_annos,
        n_cols=n_cols, 
        width=width
    )


def visualize_frames(
    frames, 
    n_cols=5, 
):
    n_frames = len(frames)
    n_rows = (n_frames - 1) // n_cols + 1
    figsize = (20, n_rows*5)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.tight_layout()
    
    for i, frame in enumerate(frames):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            ax = axes[col]
        elif n_cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f'Frame {i}')
    
    for i in range(n_frames, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            axes[col].axis('off')
        elif n_cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.show()



def get_inverse_transfrom(mean, std):
    inv_std  = [1/x for x in std]
    inv_mean = [-x for x in mean]

    return transforms.Compose(
        [
            transforms.Normalize(
                mean=[ 0., 0., 0. ],
                std=inv_std
            ),
            transforms.Normalize(
                mean=inv_mean,
                std=[ 1., 1., 1. ]
            ),
        ]
    ) 


def get_numpy_images(
    images,
    std=[0.229, 0.224, 0.225],
    mean=[0.485, 0.456, 0.406]
):
    inv_norm = get_inverse_transfrom(
        std=std, 
        mean=mean
    )
    images = inv_norm(torch.stack(images))

    all_images = []
    for img in images:
        img = (img * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
        img = np.ascontiguousarray(img)
        all_images.append(img)
        
    return all_images


def plot_clusters(
    embeddings_projected_videos,
    embeddings_projected_cluster_centers,
    labels,
    video_names,
    n_components=2,
    n_clusters=20,
    annotations=None,
    show_only_first_annotation=True,
    cluster_keywords=None,
    save_dir=None
):
    fig = go.Figure()
    for l in range(n_clusters):
        label_mask = np.array(labels) == l
        xs = embeddings_projected_videos[label_mask, 0]
        ys = embeddings_projected_videos[label_mask, 1]
        cluster_center_point = embeddings_projected_cluster_centers[l, :]
        cluster_video_names = video_names[label_mask]
        
        color = 'rgba({},{},{},{})'.format(
            int(255*np.random.rand()),
            int(255*np.random.rand()),
            int(255*np.random.rand()), 
            0.7
        )
        color_cluster_centers = 'rgba(255,0,0,1)'
        
        cluster_description_str = ''
        if cluster_keywords is not None:
            cluster_keywords_i = cluster_keywords[l]
            cluster_description_str = 'cluster tags: [ ' + ',<br>'.join(cluster_keywords_i) + ' ]'

        if annotations is not None:
            point_text = []
            for video_name in cluster_video_names:
                annos = annotations[video_name]
                anno_str = 'captions: <br>'
                if show_only_first_annotation:
                    anno_str += ' - ' + annos[0]
                else:
                    for anno in annos:
                        anno_str += ' - ' + anno + '<br>'
                point_text.append(f'{video_name} <br> {anno_str} <br><br>{cluster_description_str}')
        else:
            point_text = [f'{video_name}' for video_name in cluster_video_names]
    
        if n_components==3:
            zs = embeddings_projected_videos[label_mask, 2]
            fig.add_trace(go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode='markers',
                marker=dict(
                    color=color,
                    size=5
                ),
                text=point_text,
                name=f'cluster_{l}'
            ))
            fig.add_trace(go.Scatter3d(
                x=[cluster_center_point[0]],
                y=[cluster_center_point[1]],
                z=[cluster_center_point[2]],
                mode='markers',
                marker=dict(
                    color=color_cluster_centers,
                    size=10,
                    line=dict(
                        width=2,
                        color='DarkSlateGrey'
                    )
                ),
                text=f'cluster_{l} center',
                name=f'cluster_{l} center'
            ))
        else:
            fig.add_trace(go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                marker=dict(
                    color=color,
                    size=5
                ),
                text=point_text,
                name=f'cluster_{l}'
            ))
            fig.add_trace(go.Scatter(
                x=[cluster_center_point[0]],
                y=[cluster_center_point[1]],
                mode='markers',
                marker=dict(
                    color=color_cluster_centers,
                    size=10,
                    line=dict(
                        width=2,
                        color='DarkSlateGrey'
                    )
                ),
                text=f'cluster_{l} center',
                name=f'cluster_{l} center'
            ))
    
    fig.update_layout(
        title=go.layout.Title(
            text=f"Clustering (n_clusters={n_clusters}) results visualization",
            xref="paper",
            x=0
        ),
        autosize=False,
        width=1000,
        height=1000,
        dragmode='pan',
    )
    fig.show(config={'scrollZoom': True})

    if save_dir is not None:
        template = Template('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Plot with Images</title>
        </head>
        <body>
            <div>
                {{ plot_div }}
            </div>
        </body>
        </html>
        ''')

        # Generate the plotly div
        plot_div = fig.to_html(full_html=False)

        # Combine the plot div and the template
        html_content = template.render(plot_div=plot_div)

        # Save the HTML content to a file
        with open(save_dir / 'clustering_results.html', 'w') as file:
            file.write(html_content)


def plot_count_plot(labels):
    label_names = [f"cluster_{label}" for label in labels]
    fig = px.histogram(
        x=label_names, 
        labels={'x': 'Clusters'}, 
        title='Count plot of clusters'
    ).update_xaxes(categoryorder="total descending")
    fig.update_layout(
        xaxis_title='Labels', 
        yaxis_title='Count'
    )
    fig.show()