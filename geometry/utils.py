import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import re

# python
from typing import List, Optional, Literal


def split_cot_steps(cot_answer: str) -> List[str]:
    """
    Split CoT into steps by non-empty lines (preferred for our data format).
    If there are no newlines, fall back to a light sentence split.
    """
    text = cot_answer or ""
    # Primary: line-based split
    lines = [ln.strip() for ln in re.split(r"\r?\n", text) if ln.strip()]
    if lines:
        return lines
    # Fallback: simple sentence split
    parts = re.split(r"(?<=[\.!?])\s+|\.(?=\s|$)", text)
    return [p.strip() for p in parts if p and p.strip()]


def split_boolean_nodes(cot_text: str) -> List[str]:
    """
    Split a boolean logic CoT by **Node [XX]** markers.

    Each returned step is either a complete node block
    ('**Node [XX]**\\n* Logic: ...\\n* Result: ...') or the
    trailing '### Summary ... **Final Answer: ...**' block.
    """
    text = cot_text or ""
    parts = re.split(r"(?=\*\*Node \[\d+\]\*\*)", text)
    steps: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("**Node"):
            summary_idx = part.find("### Summary")
            if summary_idx >= 0:
                node_part = part[:summary_idx].strip()
                summary_part = part[summary_idx:].strip()
                if node_part:
                    steps.append(node_part)
                if summary_part:
                    steps.append(summary_part)
            else:
                steps.append(part)
        elif "### Summary" in part or "**Final Answer" in part:
            steps.append(part)
    return steps


def split_mcq_answer_blocks(cot_text: str) -> List[str]:
    """
    Split a structured MCQ CoT by ``Answer X:`` markers.

    Each returned step is either an answer-evaluation block
    (``Answer A:\\n* Premise: ...\\n* Reasoning: ...\\n* Conclusion: S|R``)
    or the trailing ``**Final Answer** ...`` block.
    """
    text = cot_text or ""
    parts = re.split(r"(?=Answer\s+[A-Za-z]\s*:)", text)
    steps: List[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(r"Answer\s+[A-Za-z]\s*:", part):
            final_idx = part.find("**Final Answer**")
            if final_idx >= 0:
                answer_part = part[:final_idx].strip()
                final_part = part[final_idx:].strip()
                if answer_part:
                    steps.append(answer_part)
                if final_part:
                    steps.append(final_part)
            else:
                steps.append(part)
        elif "**Final Answer**" in part or re.search(r"Answer\s*:", part):
            steps.append(part)
    return steps


def plot_trajectories_pca(
    trajectories: dict[str, list[np.ndarray]] | list[np.ndarray],
    exclude_prompt: bool = False,
    title: str = "Reasoning Trajectories (PCA)",
    save_pdf_path: Optional[str] = None,
    save_html_path: Optional[str] = None,
    width: int = 1200,
    height: int = 800,
    camera_eye: tuple[float, float, float] | None = None,
    camera_center: tuple[float, float, float] | None = None,
    camera_up: tuple[float, float, float] | None = None,
    show: bool = False,
    renderer: Optional[str] = None,
):
    """
    Project and plot reasoning trajectories in 3D PCA space.

    Accepts either:
    - A dict {label: list of embedding vectors}
    - A single list of embeddings (will be labeled 'single')
    """
    # Wrap single-answer case into dict
    if isinstance(trajectories, list):
        trajectories = {"single": trajectories}

    # Fit PCA on all points
    all_points = np.vstack([vec for embs in trajectories.values() for vec in embs])
    pca = PCA(n_components=3, random_state=42).fit(all_points)

    # Plot each trajectory
    fig = go.Figure()
    for label, embs in trajectories.items():
        if exclude_prompt:
            embs = embs[1:]
        traj = np.vstack(embs)
        traj_3d = pca.transform(traj)
        trace = go.Scatter3d(
            x=traj_3d[:, 0],
            y=traj_3d[:, 1],
            z=traj_3d[:, 2],
            mode="lines+markers",
            name=label,
            text=[f"{label} - t={i+1}" for i in range(len(traj))],
        )
        fig.add_trace(trace)

    # Layout
    fig.update_layout(
        title=title,
        width=width,
        height=height,
        scene=dict(
            xaxis_title="PC 1",
            yaxis_title="PC 2",
            zaxis_title="PC 3"
        ),
        legend_title="CoT Version"
    )

    # Optional: apply custom camera/viewpoint
    camera: dict = {}
    if camera_eye is not None:
        camera["eye"] = {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]}
    if camera_center is not None:
        camera["center"] = {"x": camera_center[0], "y": camera_center[1], "z": camera_center[2]}
    if camera_up is not None:
        camera["up"] = {"x": camera_up[0], "y": camera_up[1], "z": camera_up[2]}
    if camera:
        fig.update_layout(scene_camera=camera)

    # Optional: export to PDF
    if save_pdf_path:
        try:
            fig.write_image(save_pdf_path, format="pdf", width=width, height=height)
        except Exception as e:
            print("Failed to export PDF. Ensure 'kaleido' is installed. Error:", e)

    if save_html_path:
        try:
            fig.write_html(save_html_path)
        except Exception as e:
            print("Failed to export HTML. Error:", e)

    # Optional: display interactively (may block in headless environments)
    if show:
        try:
            if renderer is not None:
                fig.show(renderer=renderer)
            else:
                fig.show()
        except Exception as e:
            print("Failed to show figure interactively:", e)
