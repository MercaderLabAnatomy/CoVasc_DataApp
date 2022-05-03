import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pydeck as pdk
import numpy as np#Load and Cache the data
from skimage.io import imread, imshow 
from PIL import Image
from glob import glob


@st.cache(persist=True)

def getAcquiferData():
    # ACQ function to load data
    source = "Data/CovidDrugScreen_Results_MasterTable_FramerateCorrected.xlsx"
    df = pd.read_excel(source)
    df = df.drop(["Unnamed: 0",'Machine', 'Folder', 'Image_title_x','Drug_x','Drug_y', 'image_name', 'Well.1','Unnamed: 21', 'Well.2','Folder_10x', 'Heart_rate','Path_10x','Drug.1',"Hearbeat_x","Corrected_Framerate","Edema"],axis=1)
    df = df.set_index(["Drug","Experiment ID","Concentration µM","Repeat"])
    
    return df

def getDaniovisionData():
    source = "Data/CovidDrugScreen_Results_MasterTable_DanioVision.xlsx"
    df = pd.read_excel(source)
    df = df.reset_index()
    df = df.drop(["Unnamed: 0", "Folder", "File","Combo"],axis=1).reset_index()
    df = df.rename(columns= {"Concentration": "Concentration µM"})
    df=df.pivot(index=["Drug","Well","Experiment ID","Concentration µM","Replicate"], columns="Phase", values=['Distance_moved [mm]', 'Velocity [mm/s]', 'Moving [s]',
       'Not_Moving [s]'])
    #df= df.reset_index()
    df.columns=df.columns.map(('|'.join)).str.strip('|')
    return df

def getSurvivalData():
    source = "Data/CovidDrugScreen_Survival.xlsx"
    df = pd.read_excel(source)
    df_acquifer = df[df["Type"]=="Acquifer"] 
    df_daniovision = df[df["Type"]=="Daniovision"]
    return (df_acquifer,df_daniovision)

def get_literature_results():
    df_lit = pd.read_excel("Data/CovidDrugScreen_Results_Literature_intersection.xlsx")
    df_lit_counts = df_lit[df_lit.columns[[2,3,5,7,9,12,14,16]]].set_index("Tag")
    return df_lit_counts

def get_scores():
    df_ = pd.read_excel("Data/CovidDrugScreen_Score_Threshold_Kmeansclustered.xlsx").fillna(0).rename(columns={"id":"Drug"})
    df_.columns = [i.replace("Â","") for i in df_.columns]
    df_ = df_.sort_values("clusters")
    
    return df_

def survival_plot(inputdata,druglist):
    inputdata = inputdata.set_index(["Drug"]) 
    druglist=[i for i in druglist if i in list(inputdata.index)]
    
    output=inputdata.loc[druglist].reset_index()
    output["Code"]=output["Concentration in µM"].astype(str) + " µM_Replicate " + output["Replicate"].astype(str)
    data= output.pivot(columns=["Drug"],values="Survival Rate", index =["Code"]).reset_index().astype(str).set_index(["Code"])
    x_= data.columns
    y_= data.index
    
    return (data,x_,y_)    

def standardize_globalMedian(df_,collist, reference, index_columns):
    df_selcols = df_.reset_index()[collist].set_index("Drug")
    if type(index_columns) == list: 
        drop_ = ["Experiment ID"] + index_columns
        index_ = ["Experiment ID","Drug"] + index_columns
    else:
        drop_ = ["Experiment ID"] + [index_columns]
        index_ = ["Experiment ID","Drug"] + [index_columns]
        
    global_control_medians = df_selcols.drop(drop_,axis=1).loc["Control"].median()
    norm_delta = df_selcols.loc["Control"].groupby(["Experiment ID"]).median() - global_control_medians
    
    df_standardized = df_selcols.reset_index().set_index(index_) - norm_delta
    df_standardized = df_standardized.drop(index_columns,axis=1).reset_index().set_index("Drug") 
    return df_standardized    

def get_subset_results(df_,druglist):
        subset= df_.loc[druglist]
        #count_total = subset["Count_None"]
        count_others = subset["Count_None"] - subset["Count_Dev"] - subset["Count_Heart"] - subset["Count_Covid"] 
        
        count_dev = subset["Count_Dev"] - subset["Count_Heart_Dev"] - subset["Count_Dev_Covid"]

        count_heart = subset["Count_Heart"] - subset["Count_Heart_Dev"] - subset["Count_Heart_Covid"]
        
        count_covid = subset["Count_Covid"] - subset["Count_Dev_Covid"] - subset["Count_Heart_Covid"]
        
        subset_summary = pd.DataFrame({"Others":count_others, "Development": count_dev, "Cardiovascular": count_heart, "Covid": count_covid, 
        "Cardiovas.-Dev.": subset["Count_Heart_Dev"], "Covid-Dev.": subset["Count_Dev_Covid"],"Covid-Cardiovas.": subset["Count_Heart_Covid"]})
        return subset_summary.where(subset_summary > 0,0).T

def plot_pies(df_):
    row_ = int(np.ceil(len(df_.columns)/4))
    if len(df_.columns) < 4:
        col_=len(df_.columns)
    else:
        col_=4

    l = [list(li) for li in np.full((row_,col_),{'type':'domain'})]

    fig = make_subplots(rows=row_  , cols= col_, specs=l)

    i=1
    r=1
    for col in df_.columns:  
        fig.add_trace(
            go.Pie(values = df_[col], labels = df_.index,title=str((col,str(df_[col].sum())))),
            row=r, col=i)
        i+=1 
        if i > 4:
            i = 1
            r += 1
    fig.update_layout(height=600, width=1000, title_text="Literature")
    fig.update_traces(textposition='inside')

    return fig

def plot_heatmap(df_, druglist = None, show_all = True, scale = True):
    if druglist and ~show_all:
        df_ = df_.set_index("Drug").loc[druglist].reset_index()
        
    if scale:
        df_[df_.columns[2:]] = negpos_scale(df_[df_.columns[2:]])
        fig = go.Figure(data=go.Heatmap(
                        z= df_[df_.columns[2:]].T,
                        y= df_.columns[2:],
                        x= df_.Drug,
                        zmin = -1,
                        zmax = 1,
                        zmid = 0,
                        colorscale = 'PiYG'
        ))
    else:
        fig = go.Figure(data=go.Heatmap(
                    z= df_[df_.columns[2:]].T,
                    y= df_.columns[2:],
                    x= df_.Drug,
                    colorscale = 'PiYG'
        ))
    
    for cl in list(df_["clusters"].unique()):
        fig.add_vline(np.max(np.where(df_["clusters"] == cl))+0.5)
    fig.update_xaxes( tickangle = -90)
    fig.update_layout(width=1000,height=300)
    
    return fig
    
def negpos_scale(df_):
    df_scale = df_.copy()

    df_scale[df_ < 0] = df_[df_ < 0] / df_[df_ < 0].min().abs()
    df_scale[df_ > 0] = df_[df_ > 0] / df_[df_ > 0].max() 
    return df_scale
    
def get_indications():
    df_mmv = pd.read_excel("Data/CovidBox_Map.xlsx")
    df_mmv["Drug"] = df_mmv["TRIVIAL_NAME"] + "_" + df_mmv["Plate ID"].astype(str) + df_mmv["Well ID"]
    df_mmv = df_mmv[["TRIVIAL_NAME","Drug","Indication"]]
    df_mmv[["Indication 1","Indication 2"]] = df_mmv["Indication"].str.split(" - ",expand = True)
    df_mmv = df_mmv.drop("Indication",axis=1)
    ind1 = list(df_mmv["Indication 1"].unique())
    ind2 = list(df_mmv["Indication 2"].unique())
    return df_mmv, ind1, ind2

def authentication():
    name_c = st.secrets["DB_TOKEN"]
    username_c = st.secrets["DB_USERNAME"]
    password_c = st.secrets["DB_PASSWORD"]
    authentication_status = None
    
    with st.form(key="Login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.form_submit_button("Login")
        if (username == username_c) & (password == password_c): 
            authentication_status = True
            st.write("Login successful")
        elif (username == "") | (password == ""):
            st.write("Enter a username and password")
        else:
            authentication_status = False
            st.write("Username/Password wrong")
    return (authentication_status, username)
    
st.set_page_config(  layout="wide")

st.title('Interactive Data App - Covid Box Drug Screening')


st.subheader('Workflow Scheme')

image = Image.open('Data/Images/Design/Online_Short workflow.png')

st.image(image) 


df_indications, i1, i2 = get_indications()

df1 = getAcquiferData()
druglist = df1.index.levels[0].tolist()
druglist1 = ["Remdesivir_1A06","Hydroxychloroquine_1F11","Lopinavir_1H04","Ritonavir_1A11","Favipiravir_1D10","Ivermectin_1F05","Ribavirin_1F08"
,"Umifenovir_2E02","Baricitinib_2C09","Mycophenolic acid_2B10"]

with st.sidebar:
    a_state,user = authentication()

if a_state:

    #Sidebar
    st.sidebar.title('Choose your compounds of interest')

    x = st.sidebar.multiselect('Choose Compounds', druglist, druglist1)
    ind_primary = st.sidebar.multiselect('Optional: Choose by primary drug indication', i1, [])
    ind_secondary = st.sidebar.multiselect('Optional: Choose by secondary drug indication', i2, [])

    if ind_primary:
        x.clear()
        x = list(df_indications.set_index('Indication 1').loc[ind_primary]["Drug"].values)

    conc05 = st.sidebar.checkbox('Show also Concentration 0.5 µM')

    if conc05:
         st.write('Showing 0.5 µM and 1 µM treatment concentration. Only a subset of compounds were tested at 0.5 µM.')

    x_insert = x.copy()
    xlength = len(x)
    experiment_ids = np.sort(df1.loc[x].reset_index()["Experiment ID"].unique())
    x_insert.insert(0,'Control')

    with st.expander("See heatmap"):
        df_test = get_scores()
        fig_test = plot_heatmap(df_test, druglist = x, show_all = False, scale = True)
        st.plotly_chart(fig_test,use_container_width=True)
        

    with st.expander("See literature counts"):
        
        df_lit_search = get_literature_results()
        df_out = get_subset_results(df_lit_search, x)
        fig_pie = plot_pies(df_out)
        st.plotly_chart(fig_pie,use_container_width=True)
        
    with st.expander("See survival"):
        df_survival_ACQ,df_survival_DV = getSurvivalData()   
        cols_survival_ACQ = st.container() 
        
        with cols_survival_ACQ:
            data_ACQ,x_ACQ,y_ACQ = survival_plot(df_survival_ACQ, x)
            fig_survival_ACQ = px.imshow(data_ACQ,
                        labels=dict(x="Drug", y=("4 dpf with PTU"), color="Survival"),
                        x=x_ACQ,
                        y=y_ACQ,
                        color_continuous_scale='Spectral',
                        zmin=0,
                        zmax=100
                        )
            fig_survival_ACQ.update_xaxes(tickangle=45)
            st.plotly_chart(fig_survival_ACQ,use_container_width=True)

        cols_survival_DV = st.container() 
        with cols_survival_DV:
            data_DV,x_DV,y_DV = survival_plot(df_survival_DV, x)
            fig_survival_DV = px.imshow(data_DV,
                        labels=dict(x="Drug", y=("5 dpf no PTU"), color="Survival"),
                        x=x_DV,
                        y=y_DV,
                        color_continuous_scale='Spectral',
                        zmin=0,
                        zmax=100
                        )
            fig_survival_DV.update_xaxes(tickangle=45)
            st.plotly_chart(fig_survival_DV,use_container_width=True)

    with st.expander("See morphological analysis"):
        standardize = st.checkbox('Standardize morphological data to global control median')
        groupbydrug = st.checkbox('Group morphological graphs by drug')
        
        if standardize:
            collist1 = ['Drug','Experiment ID','Concentration µM','Length','N_ISV','Median_minor_axis_length','Median_major_axis_length','Delta_DIA-SYS','Heart_BPM']
            df1 = standardize_globalMedian(df1 ,collist1, "Control", "Concentration µM")
        
        df_selected = df1.loc[x_insert].reset_index()
        plotdata = df_selected.set_index("Experiment ID").loc[experiment_ids].reset_index()
        plotdata["Experiment ID"] = plotdata["Experiment ID"].astype(str)
        
        if not conc05:
            plotdata= plotdata[plotdata["Concentration µM"]==1.0]
           
        
            
        # ACQ Choose the measurement to plot
        measurement = st.selectbox('Toggle between ', ('Length', 'Heart_BPM','Median_minor_axis_length','Median_major_axis_length','N_ISV','Delta_DIA-SYS'))
        
        # ACQ Plot the selected measurement 
        if groupbydrug:
            fig = px.violin(plotdata, y=measurement, x="Drug", color="Drug",facet_col="Concentration µM",category_orders={"Drug":x_insert}, box=True, points="all", hover_data=plotdata.columns)
            st.plotly_chart(fig,use_container_width=True)
            
        else: 
            fig = px.violin(plotdata, y=measurement, x="Experiment ID", color="Drug",facet_col="Concentration µM",category_orders={"Drug":x_insert}, box=True, points="all", hover_data=plotdata.columns)
            st.plotly_chart(fig,use_container_width=True)

    with st.expander("See activity analysis"):
        df2 = getDaniovisionData()
        measurement2 = st.selectbox('Toggle between ', df2.columns)
        standardize2 = st.checkbox('Standardize activity data to global control median')
        groupbydrug2 = st.checkbox('Group activity graphs by drug')
        
        if standardize2:
            collist2 = list(df2.reset_index().columns)
            df2 = standardize_globalMedian(df2 ,collist2, "Control", ["Concentration µM","Replicate"])
        
        experiment_ids2=np.sort(df2.loc[x].reset_index()["Experiment ID"].unique())
        df_selected2=df2.loc[x_insert].reset_index()
        
        
        plotdata2=df_selected2.set_index("Experiment ID").loc[experiment_ids2].reset_index()
        plotdata2["Experiment ID"] = plotdata2["Experiment ID"].astype(str)

        if not conc05:
            plotdata2= plotdata2[plotdata2["Concentration µM"]==1.0]
        
        if groupbydrug2:
            fig2 = px.violin(plotdata2, y=measurement2, x="Drug", color="Drug",facet_col="Concentration µM",category_orders={"Drug":x_insert}, box=True, points="all",hover_data=plotdata2.columns)
            st.plotly_chart(fig2,use_container_width=True)

        else:
            fig2 = px.violin(plotdata2, y=measurement2, x="Experiment ID", color="Drug",facet_col="Concentration µM",category_orders={"Drug":x_insert}, box=True, points="all",hover_data=plotdata2.columns)
            st.plotly_chart(fig2,use_container_width=True)

    with st.expander("See images of larvae"):
        # ACQ Show images from the acquifer for each compound
        ids=plotdata["Experiment ID"].unique()
        plotimages=plotdata.set_index("Experiment ID").loc[ids[0]]["Drug"].unique()
        listexperiments={expid: plotdata.set_index("Experiment ID").loc[expid]["Drug"].unique() for expid in ids }
        cols = st.columns(len(listexperiments))

        for id,(col,experiment) in enumerate(zip(cols,listexperiments)):
            st.header('Experiment ID: '+ experiment)
            
            compoundsexperiment = list( listexperiments[experiment] )
            compoundsexperiment.remove("Control")
            
            expcols = st.columns(len(compoundsexperiment))
            
            Control_BF=glob("Images/Example images/2x_examples/Experiment_ID_"+experiment+"/Control_C1_*.jpg")[0]
            Control_GFP=glob("Images/Example images/2x_examples/Experiment_ID_"+experiment+"/Control_C2_*.jpg")[0]
            
            for drugcol,drug in zip(expcols,compoundsexperiment): 
                
                st.header('Control vs. ' + drug)
                
                drug_BF=glob("Images/Example images/2x_examples/Experiment_ID_"+experiment+"/" + drug + "_C1_*.jpg")[0]
                drug_GFP=glob("Images/Example images/2x_examples/Experiment_ID_"+experiment+"/" + drug + "_C2_*.jpg")[0]
                
                
                cols1,cols2,cols3,cols4 = st.columns(4)
                
                with cols1:
                    im1=Image.open(Control_BF)
                    st.image(im1, caption='2x-BF', width=None, use_column_width=None, clamp=True, channels="RGB", output_format="auto")
                    
                with cols2:
                    im2=Image.open(Control_GFP)
                    st.image(im2, caption='2x-Fli:GFP', width=None, use_column_width=None, clamp=True, channels="RGB", output_format="auto")
                    
                with cols3:
                    im3=Image.open(drug_BF)
                    st.image(im3, caption='2x-BF', width=None, use_column_width=None, clamp=True, channels="RGB", output_format="auto")
                    
                with cols4:
                    im4=Image.open(drug_GFP)
                    st.image(im4, caption='2x-Fli:GFP', width=None, use_column_width=None, clamp=True, channels="RGB", output_format="auto")
                    
            
            #im=(im/np.max(im))*255
            #im = Image.fromarray(im[0])
            
            #st.image("https://static.streamlit.io/examples/cat.jpg")