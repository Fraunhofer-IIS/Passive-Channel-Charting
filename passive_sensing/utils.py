import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from sklearn.manifold import trustworthiness
from scipy.spatial import distance_matrix
import numpy as np


def least_squares_transform(cc_data, pos_data):

    # Pad the data with ones, so that our transformation can do translations too
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    X = pad(cc_data)
    Y = pad(pos_data)

    A, _, _, _ = np.linalg.lstsq(X, Y)

    transform = lambda x: unpad(np.dot(pad(x), A))

    return transform

def make_cols(xs: np.ndarray, ys: np.ndarray):

    color1 = (xs - min(xs)) / max(xs - min(xs))
    color2 = (ys - min(ys)) / max(ys - min(ys))
    color3 = np.zeros(len(color2))
    return np.concatenate([color1[:, None], color2[:, None], color3[:, None]], axis=-1)

def make_cols_rgb(xs: np.ndarray, ys: np.ndarray):
    color1 = (xs - min(xs)) / (max(xs) - min(xs))
    color2 = (ys - min(ys)) / (max(ys) - min(ys))
    color3 = np.zeros(len(color2))
    colors = np.concatenate([color1[:, None], color2[:, None], color3[:, None]], axis=-1)
    
    # Convert normalized colors to RGB strings
    color_strings = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    return color_strings

def plot_channel_chart(ue_xs, ue_ys, embedding_xs, embedding_ys, cc_suptitle="", path=None, ax=plt.gca()):
    c = make_cols(ue_xs, ue_ys)

    ax.scatter(embedding_xs, embedding_ys, c=c, marker='o', s=4., rasterized=True)
    ax.set_title(cc_suptitle)
  
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")

    if path is not None:
        plt.savefig(path, bbox_inches='tight')     


def plot_channel_chart_plotly(ue_xs, ue_ys, embedding_xs, embedding_ys):
    assert len(ue_xs) == len(ue_ys) and ue_xs.shape == ue_ys.shape and len(ue_ys.shape) == 1
    assert embedding_xs.shape == ue_xs.shape and embedding_ys.shape == ue_ys.shape

    c = make_cols_rgb(ue_xs, ue_ys)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=embedding_xs,
        y=embedding_ys,
        mode='markers',
        marker=dict(
            color=c,
            size=4,
        )
    ))

    fig.update_layout(
        xaxis_title="x [m]",
        yaxis_title="y [m]",
        template="plotly"
    )

    return fig


# Continuity is identical to trustworthiness, except that original space and embedding space are swapped
def continuity(*args, **kwargs):
	args = list(args)
	args[0], args[1] = args[1], args[0]
	return trustworthiness(*args, **kwargs)

def kruskal_stress(X, X_embedded):
	dist_X = distance_matrix(X, X)
	dist_X_embedded = distance_matrix(X_embedded, X_embedded)
	beta = np.divide(np.sum(dist_X * dist_X_embedded), np.sum(dist_X_embedded * dist_X_embedded))

	return np.sqrt(np.divide(np.sum(np.square((dist_X - beta * dist_X_embedded))), np.sum(dist_X * dist_X)))