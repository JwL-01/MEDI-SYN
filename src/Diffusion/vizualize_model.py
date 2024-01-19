from graphviz import Digraph

def plot_custom_model_side_by_side():
    dot = Digraph(comment='Stepwise Reverse Diffusion Network', format='png')
    dot.attr(rankdir='LR')  # Left to Right layout for the main graph
    dot.attr('node', shape='box')  # Set the default shape to box for all nodes

    # Define subgraphs for the encoder and decoder, with a Top to Bottom layout within each
    with dot.subgraph(name='cluster_encoder') as enc:
        enc.attr(rankdir='TB')  # Top to Bottom layout for the encoder
        enc.attr(label='Encoder')
        enc.node('E1', 'EncoderBlock 1\nConv2D(128->64)\nBatchNorm2D\nSiLU\nSE-Attention')
        enc.node('E2', 'EncoderBlock 2\nConv2D(64->32)\nBatchNorm2D\nSiLU\nSE-Attention')
        enc.node('E3', 'EncoderBlock 3\nConv2D(32->16)\nBatchNorm2D\nSiLU\nSE-Attention')
        enc.node('E4', 'EncoderBlock 4\nConv2D(16->8)\nBatchNorm2D\nSiLU\nSE-Attention')
        enc.edges([('E1', 'E2'), ('E2', 'E3'), ('E3', 'E4')])

    with dot.subgraph(name='cluster_decoder') as dec:
        dec.attr(rankdir='TB')  # Top to Bottom layout for the decoder
        dec.attr(label='Decoder')
        dec.node('D1', 'DecoderBlock 1\nConvTrans2D(8->16)\nBatchNorm2D\nSiLU\nSE-Attention')
        dec.node('D2', 'DecoderBlock 2\nConvTrans2D(16->32)\nBatchNorm2D\nSiLU\nSE-Attention')
        dec.node('D3', 'DecoderBlock 3\nConvTrans2D(32->64)\nBatchNorm2D\nSiLU\nSE-Attention')
        dec.node('D4', 'DecoderBlock 4\nConvTrans2D(64->128)\nBatchNorm2D\nSiLU\nSE-Attention')
        dec.edges([('D1', 'D2'), ('D2', 'D3'), ('D3', 'D4')])

    # Nodes for input, FiLM, and output
    dot.node('I', 'Input\n1x128x128',shape='parallelogram')
    dot.node('F', 'FiLM\n512')
    dot.node('O', 'Output\nTanh')

    # Nodes for time representation
    dot.node('T', 'Time\nRepresentation', shape='parallelogram')

    # Edges between encoder, FiLM, and decoder
    dot.edge('E4', 'F')
    dot.edge('F', 'D1')

    # Skip connections
    dot.edge('E1', 'D3', style='dashed')
    dot.edge('E2', 'D2', style='dashed')
    dot.edge('E3', 'D1', style='dashed')

    # Time Representation connection and output
    dot.edge('T', 'F', style='dashed')
    dot.edge('D4', 'O')

    # Save and render the graph
    dot.render('imaging/model_vizualization', format='eps')

plot_custom_model_side_by_side()
