from jinja2 import Template
import numpy as np
# this creates the feature extractor 
def write_df(df):
    inds = range(len(df))
    already_done_list = []
    output = []
    
    for j in df.iloc[inds].iterrows():
        feature_index = j[0]
        atominds = np.array(j[1]["atominds"])
        resids = j[1]["resids"]
        feat = j[1]["featuregroup"]
        func = get_feature_function(df, feature_index)
        
        feat_label = feat+"_%s"%'_'.join(map(str,resids))

        if feat_label not in already_done_list:
            #mdtraj is 0 indexed and plumed is 1 indexed
            output.append(func(atominds + 1 , feat_label))
            already_done_list.append(feat_label)
            
    for j in df.iloc[inds].iterrows():
        feature_index = j[0]
        atominds = j[1]["atominds"]
        feat = j[1]["featuregroup"]
        resids = j[1]["resids"]
        feat = j[1]["featuregroup"]        
        argument = feat+"_%s"%'_'.join(map(str,resids))
        label = "l0%d"%feature_index
        
        output.append(create_feature(argument, df.otherinfo[feature_index], label))
        output.append("\n")
    
    return ''.join(output)

# this creates a fully connected layer
def render_fc_layer(layer_indx, lp):
    output=[]
    for i in np.arange(lp.out_features):
        
        arg=','.join(["l%d%d"%(layer_indx-1,j) for j in range(lp.in_features)])
        
        weights = ','.join(map(str,lp.weight[i].data.tolist()))
        bias =','.join(map(str,lp.bias[i].data.tolist()))
        
        # combine without bias
        non_bias_label = "l%d%dnb"%(layer_indx, i)
        output.append(plumed_combine_template.render(arg = arg,
                                   coefficients = weights,
                                   label=non_bias_label,
                                   periodic="NO") +"\n")
        # now add the bias
        bias_label = "l%d%d"%(layer_indx, i)
        output.append(create_neural_bias(non_bias_label, bias, bias_label))
        output.append("\n")
    return ''.join(output)
    
    
# this cretes a sigmoid layer
def render_sigmoid_layer(layer_indx, lp, hidden_size=50):
    output=[]    
    for i in np.arange(hidden_size):
        arg="l%d%d"%(layer_indx-1, i)
        label = "l%d%d"%(layer_indx, i)
        output.append(create_sigmoid(arg, label))
        output.append("\n")
        
    return ''.join(output)


def render_network(net):
    output =[]
    # Start by evaluating the actual dihedrals + sin-cosine transform aka the input features 
    output.append(write_df(net.df))
    index = 0
    # Go over every layer of the netowrk
    for lp in net.children():
        index += 1
        if str(lp).startswith("Linear"):
            output.append(render_fc_layer(index, lp))
        elif str(lp).startswith("Sigmoid"):
            output.append(render_sigmoid_layer(index, lp,hidden_size=net.hidden_size))
        else:
            raise ValueError("Only Linear and Sigmoid Layers are supported for now")
    # Lastly, we want to print out the values from the last layer. This becomes our CV. 
    arg = "l%d0"%index
    output.append(render_print_val(arg))
    return ''.join(output)


plumed_torsion_template = Template("TORSION ATOMS={{atoms}} LABEL={{label}} ")

plumed_matheval_template = Template("MATHEVAL ARG={{arg}} FUNC={{func}} LABEL={{label}} PERIODIC={{periodic}} ")

plumed_combine_template = Template("COMBINE LABEL={{label}} ARG={{arg}} COEFFICIENTS={{coefficients}} "+\
                                    "PERIODIC={{periodic}} ")
plumed_print_template = Template("PRINT ARG={{arg}} STRIDE={{stride}} FILE={{file}} ")


def create_torsion_label(inds, label):
    #t: TORSION ATOMS=inds
    return plumed_torsion_template.render(atoms=','.join(map(str, inds)), label=label) +"\n"


def create_feature(argument, func, feature_label):
    arg = argument
    x="x"
    if func in ["sin","cos"]:
        f = "%s(%s)"%(func,x)
        label = feature_label
    else:
        raise ValueError("Can't find function")

    return plumed_matheval_template.render(arg=arg, func=f,\
                                           label=label,periodic="NO")


def create_neural_bias(nb, bias, label):
    arg = ",".join([nb])
    f = "+".join(["x", bias])
    return plumed_matheval_template.render(arg=arg, func=f,\
                                           label=label,periodic="NO")
def create_sigmoid(arg, label):
    f = "1/(1+exp(-x))"
    return plumed_matheval_template.render(arg=arg, func=f,\
                                           label=label,periodic="NO")

def render_print_val(arg,stride=1,file="CV"):
    return plumed_print_template.render(arg=arg,
                                       stride=stride,
                                       file=file)
def get_feature_function(df, feature_index):
    possibles = globals().copy()
    possibles.update(locals())
    func = possibles.get("create_torsion_label")
    return func
def match_mean_free_function(df, feature_index):
    func = df.otherinfo[feature_index]
    return func

