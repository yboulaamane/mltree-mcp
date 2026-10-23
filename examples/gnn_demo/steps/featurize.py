"""Step 2 - turn SMILES into molecular graphs (node features + edge list).

Uses RDKit when it is installed; otherwise a tiny regex tokenizer that is good
enough for the toy dataset (it handles chains, branches, ring closures and
aromatic lowercase atoms, but not stereo or charges).
"""
import re

from _common import OUT, load, save

ELEMENTS = ["C", "N", "O", "Cl", "S", "F"]


def one_hot(sym: str, aromatic: bool) -> list[float]:
    v = [1.0 if sym == e else 0.0 for e in ELEMENTS]
    return v + [1.0 if aromatic else 0.0]


def toy_graph(smiles: str) -> dict:
    tokens = re.findall(r"Cl|Br|[BCNOSPF]|[cnos]|\(|\)|=|#|\d", smiles)
    nodes, edges, stack, rings = [], [], [], {}
    prev, order = None, 1
    for t in tokens:
        if t == "(":
            stack.append(prev)
        elif t == ")":
            prev = stack.pop()
        elif t in "=#":
            order = 2 if t == "=" else 3
        elif t.isdigit():
            if t in rings:
                edges.append((rings.pop(t), prev, 1))
            else:
                rings[t] = prev
        else:
            idx = len(nodes)
            nodes.append(one_hot(t.capitalize() if t != "Cl" else t, t.islower()))
            if prev is not None:
                edges.append((prev, idx, order))
            prev, order = idx, 1
    return {"x": nodes, "edges": edges}


def rdkit_graph(smiles: str) -> dict:
    from rdkit import Chem  # noqa: PLC0415

    mol = Chem.MolFromSmiles(smiles)
    x = [one_hot(a.GetSymbol(), a.GetIsAromatic()) for a in mol.GetAtoms()]
    e = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())) for b in mol.GetBonds()]
    return {"x": x, "edges": e}


try:
    import rdkit  # noqa: F401

    to_graph = rdkit_graph
except ImportError:
    to_graph = toy_graph

graphs = []
for m in load("molecules.json"):
    g = to_graph(m["smiles"])
    g.update(smiles=m["smiles"], y=m["y"])
    graphs.append(g)
save("graphs.json", graphs)
print(f"featurized {len(graphs)} graphs with {to_graph.__name__}")

try:  # also write a PyTorch Geometric dataset when torch_geometric is installed
    import torch
    from torch_geometric.data import Data, InMemoryDataset
except ImportError:
    pass
else:
    def to_data(g: dict) -> Data:
        pairs = [(a, b) for a, b, _ in g["edges"]] + [(b, a) for a, b, _ in g["edges"]]
        edge_index = torch.tensor(pairs, dtype=torch.long).t().reshape(2, -1)
        return Data(x=torch.tensor(g["x"]), edge_index=edge_index, y=torch.tensor([g["y"]]))

    data, slices = InMemoryDataset.collate([to_data(g) for g in graphs])
    torch.save((data.to_dict(), slices), OUT / "graphs.pt")
    print("wrote data/processed/graphs.pt (PyG)")
