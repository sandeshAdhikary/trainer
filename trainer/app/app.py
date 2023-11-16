import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from trainer.utils import pretty_title
from trainer.study import Study
from trainer.app import app_utils
import pandas as pd
import json
import altair as alt
from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode, DataReturnMode
from trainer.app.app_utils import exclude_key

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


if __name__ == "__main__":
    st.set_page_config(layout="wide") 
    st.header("Study Analysis")
    with st.sidebar:
        with st.form(key='config_form') as user_config_form:
            study_cfg =  app_utils.load_config()
    if study_cfg is None:
        st.write("Enter the path (in the sidebar) to a study's yaml config-file to analyze it.")
    if study_cfg is not None:

        study = Study(study_cfg)
        runs = study.show_runs()

        # Title
        st.title(f"{pretty_title(study.name)} Study")
        st.markdown(study.config['desc'])

        # Study Information
        with st.expander("View the study's default config", expanded=False):
            app_utils.study_defaults(study)

        # Select Runs to Plot
        st.header("Select Runs")
        with st.expander("Select Runs", expanded=True):
            st.write("""Check runs in the `Select` column to include runs. 
                     Use the `Tag` column to create new groups.
                     Use the Group Sselector in the sidebar to define groups.
                     """)
            runs = run_selector(study)
            if runs is not None:
                # Training metrics
                all_train_data = []
                not_found = []
                for idr, row in runs.iterrows():
                    project = row['project']
                    sweep = row['sweep']
                    run_id = row['run_id']
                    run_group =  f"Project: {project} | Sweep: {sweep} | Run: {run_id}"
                    if sweep is not None:
                        train_file = f'{project}/sweep_{sweep}/{run_id}/eval/train_history.json'
                    else:
                        train_file = f'{project}/{run_id}/eval/train_history.json'
                    try:
                        train_data = study.storage.load(train_file, filetype='json')
                        if isinstance(train_data, str):
                            train_data = json.loads(train_data)
                        train_data = pd.DataFrame(train_data)
                        train_data['run_group'] = run_group
                        all_train_data.append(train_data)
                    except FileNotFoundError:
                        not_found.append(run_group)
                if len(all_train_data) > 0:
                    all_train_data = pd.concat(all_train_data)

                
                    smoothing_value = st.slider("Smoothing", min_value=0, max_value=50, step=1, value=1,)
                    smoothing_value = -smoothing_value

                    selection = alt.selection_point(fields=['run_group'], bind='legend')
                    main_chart = alt.Chart(all_train_data, title='Training Summary').mark_line(opacity=0.1).encode(
                        x=alt.X('trainer_step:Q', title='Training Steps'),
                        y=alt.Y('train/episode_reward:Q', title='Episode Reward'),
                        color=alt.Color('run_group:N'),
                        tooltip='run_group:N',
                    ).add_params(
                        selection
                    )
                    smooth_chart = alt.Chart(all_train_data, title='Training Summary').mark_line().transform_window(
                        rolling_mean='mean(train/episode_reward)',
                        frame=[smoothing_value, 0]).encode(
                        x=alt.X('trainer_step:Q', title='Training Steps'),
                        y=alt.Y('rolling_mean:Q', title='Episode Reward'),
                        color=alt.Color('run_group:N'),
                        tooltip='run_group:N',
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.2))
                    ).add_params(
                        selection
                    )
                    chart = main_chart + smooth_chart


                    st.altair_chart(chart, use_container_width=True)
                if len(not_found) > 0:
                    st.warning(f"""Could not find training history for the following runs:""")
                    st.write(not_found)


        if runs is not None:

            # Define columns to group-by
            with st.sidebar:
                runs, group_cols = app_utils.group_runs(runs)
                runs = app_utils.avg_over_group(runs)


            # View selected data
            with st.expander("View Data", expanded=False):
                app_utils.view_selected_data(runs)
            
            # Plot metrics
            screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR_WIDTH')
            container_width = screen_width

            # Evaluation metrics
            st.header("Evaluation Metrics")
            for metric_name, metric in study.metrics.items():
                    with st.expander(pretty_title(metric_name), expanded=False) as container:
                        # Slider to select chart size
                        chart_size = st.slider("Chart Size", min_value=0.1, max_value=1.0, value=0.3,
                                                key=f"chart_size_{metric_name}", format="")
                        chart_size = container_width*chart_size
                        # Get metric data
                        metric_data = study.metric_table(metric_name, limit=None)
                        metric_data = metric_data.merge(runs, on=['run_id', 'sweep', 'project'],how='right')
                        metric_data = metric_data.dropna(subset=['eval_name'])                        
                        # Average out data over the groups                        
                        # metric_data = app_utils.avg_over_group(metric_data)
                        # Plots
                        if metric_data.shape[0] > 0:
                            st.subheader(pretty_title(metric_name))
                            if metric_name == 'observation_videos':
                                app_utils.plot_videos(metric_data, storage=study.storage)
                            else:
                               facet = {'name': 'eval_name', 'columns': int(container_width//chart_size)}
                               app_utils.plot_scalars(metric, metric_data, group_cols, chart_size, facet)






    