import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode, DataReturnMode
from omegaconf import OmegaConf
from trainer.utils import pretty_title


def plot_scalars(metric, data, group_cols, chart_size, facet=None):
    if facet is None:
        facet = {}
    chart, chart_info = metric.plot(data, facet=facet, 
                                    show_legend=False,
                                    chart_properties={'width': chart_size, 
                                                    'height': chart_size,}
                                                    )
    legend_table = make_legend_table(data, group_cols, 
                                     legend_color_map=chart_info['legend_colors'], 
                                     legend_col='group')
    st.dataframe(legend_table, hide_index=True)
    st.altair_chart(chart)

def plot_videos(data, storage):
    project_select = st.selectbox("Select Project", data['project'].unique())
    sweep_select = st.selectbox("Select Sweep", data.query(f"project=='{project_select}'")['sweep'].unique())
    run_select = st.selectbox("Select Run", data.query(f"project=='{project_select}' and sweep=='{sweep_select}'")['run_id'].unique())
    evals = data.query(f"project=='{project_select}' and sweep=='{sweep_select}' and run_id=='{run_select}'")
    num_vids = evals.shape[0]
    vid_cols = st.columns(num_vids)
    for idv in range(num_vids):
        data_row = evals.iloc[idv]
        filepath = data_row['filepath']
        filepath = filepath.split(storage.dir)[1].lstrip('/')
        video = storage.load(filepath, filetype='bytesio')
        vid_cols[idv].video(data=video)
        vid_cols[idv].write(f"Eval Name: `{pretty_title(data_row['eval_name'])}`")

def make_legend_table(data, group_cols, legend_color_map, legend_col):
    cols = [*group_cols, 'group']
    data = data.loc[:, cols].drop_duplicates()
    # legend_table = data.loc[:,cols].drop_duplicates()
    data.rename({f'{legend_col}': "legend"}, axis='columns', inplace=True)
    # move legend to first column
    col = data.pop("legend")
    data.insert(0, col.name, col)
    data.reset_index(drop=True, inplace=True)
    data.columns = [pretty_title(x) for x in data.columns]
    data = data.style.map(lambda x: df_apply_colors(x, legend_color_map), subset=['Legend'])
    return data
    
def df_apply_colors(val, color_dict):
     color = color_dict[val]
     return f'background-color: {color}; color: {color}; font-size:0.01em'

def run_selector(study):

    # Get list of runs
    runs = study.show_runs()
    

    all_dicts = []
    all_keys = []
    for idr, row in runs.iterrows():
        # Get run info from study
        project, sweep, run_id, steps = row['project'], row['sweep'], row['run_id'], row['steps']
        if sweep.lower() == 'none':
            sweep = None
        run_info = study.run_info(project=project, sweep=sweep, run_id=run_id)
        
        # Reformat model and trainer dicts as sets
        model_dict, trainer_dict = {}, {}
        for k in run_info['model'].keys():
            model_dict[k] = str(run_info['model'][k])
        for k in run_info['trainer'].keys():
            trainer_dict[k] = str(run_info['trainer'][k])
        model_dict = set(model_dict.items())
        trainer_dict = set(trainer_dict.items())
        info_dict = set([('run_id', run_id), ('sweep', sweep), ('steps', steps)])
        run_dict = set.union(model_dict, trainer_dict, info_dict)
        all_dicts.append(run_dict)
        all_keys.append(set([x[0] for x in run_dict]))
    
    # Combine all sets, only keep keys that are different
    all_keys = set.union(*all_keys)
    matching_keys =  set([x[0] for x in set.intersection(*all_dicts)])
    diff_keys = all_keys - matching_keys
    diff_keys = [x for x in diff_keys if not exclude_key(x)]

    if 'project' not in diff_keys:
        diff_keys.append('project')
    if 'sweep' not in diff_keys:
        diff_keys.append('sweep')

    for idx in range(len(all_dicts)):
         all_dicts[idx] = {k:v for k,v in all_dicts[idx] if k in diff_keys}
    df = pd.DataFrame(all_dicts)    
    df['tag'] = [None]*len(df)

    new_cols = [x for x in df.columns if x not in runs.columns]
    df = df[['project', 'sweep', 'run_id', 'steps', *new_cols]]

    builder = GridOptionsBuilder.from_dataframe(df)
    builder.configure_column(field='sweep', width=100, header_name="Sweep")
    builder.configure_column(field='project', width=200, header_name="Project")
    builder.configure_column(field='run_id', width=100, header_name="Run")
    builder.configure_column(field='steps', width=100, header_name="Steps")
    builder.configure_column(field='tag', width=100, header_name="Tag", editable=True)
    builder.configure_column(field='select', checkboxSelection=True, header_name="Select")

    builder.configure_grid_options(groupDefaultExpanded=-1, rowSelection='multiple',)
    gridOptions = builder.build()
    # gridOptions['getRowStyle'] = jscode
    data_selector = AgGrid(df, gridOptions=gridOptions,
                columns_auto_size_mode=1,theme="streamlit", allow_unsafe_jscode=True,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                header_checkbox_selection_filtered_only=True
    )
                # update_mode=GridUpdateMode.MODEL_CHANGED,)
    # Get filtered data
    final_data = pd.DataFrame(data_selector.data).fillna("None")
    if len(data_selector.selected_rows) > 0:
        selected_data = pd.DataFrame(data_selector.selected_rows).drop('_selectedRowNodeInfo', axis='columns')
        selected_data = selected_data.fillna("None")
        final_data = final_data.merge(selected_data, how='inner', on=final_data.columns.tolist())
        final_data['sweep'] = final_data['sweep'].apply(lambda x: str(x).lower())
        return final_data
    return None

def exclude_key(x):
    exclude = False
    exclude = exclude or x.startswith('storage')
    exclude = exclude or ('freq' in x)
    exclude = exclude or (x == 'env')
    return exclude

def study_defaults(study):
    for key, value in study.config.items():
        if key != 'desc':
            st.write(pretty_title(key))
            if isinstance(value, dict):
                st.json(value, expanded=False)
            else:
                st.write(f"`{value}`")

def group_runs(runs, default_group_cols=None):
    if default_group_cols is None:
         default_group_cols = ['project', 'sweep']
    group_cols = st.multiselect(label='Select Group Columns', 
                                options=runs.columns, default=default_group_cols, 
                                placeholder="Select Group Columns")
    if len(group_cols) == 0:
        # Use run_id as group
        runs['group'] = runs.apply(lambda x: f"run_{x['run_id']}", axis=1)
        group_cols = ['run_id']
    else:
        # Group by groups provided
        runs['group'] = runs.groupby(group_cols).ngroup()
        runs['group'] = runs.apply(lambda x: ' | '.join([f'{g} = {x[g]}' for g in group_cols]), axis=1)
    return runs, group_cols

def avg_over_group(data, group_by_cols=None):
    """"
    data should have a column named 'group'
    will return a dataframe grouped by the 'group' col
    all numerical columns will be averaged
    For non-numeric columns, we'll take the first value
    """
    if group_by_cols is None:
        group_by_cols  = ['group']
    if 'step' in data.columns:
        group_by_cols.append('step')
    # Define custom aggregation functions
    aggregation_funcs = {}
    # Get the list of numeric columns
    numeric_columns = data.select_dtypes(include=['number']).columns
    # Get the list of non-numeric columns
    non_numeric_columns = data.select_dtypes(exclude=['number']).columns
    # Apply 'mean' aggregation to numeric columns
    for col in numeric_columns:
        aggregation_funcs[col] = 'mean'

    # Apply 'first' aggregation to non-numeric columns
    for col in non_numeric_columns:
        aggregation_funcs[col] = 'first'

    result = data.groupby(group_by_cols).agg(aggregation_funcs).reset_index(drop=True)
    
    return result


def load_config():
    config_path = st.text_input("Enter config path", value=None)
    config_button = st.form_submit_button(label='Analyze')
    config = None
    if config_path is not None:
        try:
            config = OmegaConf.load(config_path)
            return OmegaConf.to_container(config, resolve=True)
        except (FileNotFoundError, IsADirectoryError):
            st.error(f"Config file {config_path} not found")
    return config


def view_selected_data(runs):
        st.write("""
                Here's the data based on your selection and grouping criteria.
                The `group` column shows each run's assigned group.
                Numeric values have been averaged over the groups. For non-numeric 
                columns, we display the first value for the group.
                """)
        projects = ' '.join([f"`{x}`" for x in runs['project'].unique()])
        st.write(f"Projects: {projects}")
        sweeps = ' '.join([f"`{x}`" for x in runs['sweep'].unique()])
        st.write(f"Sweeps: {sweeps}")
        runs = runs[['group', *runs.columns.difference(['group'])]]
        st.write(runs)