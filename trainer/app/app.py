import streamlit as st
from omegaconf import OmegaConf
import os
import altair as alt
import pandas as pd
from streamlit_js_eval import streamlit_js_eval
from trainer.utils import pretty_title, COLORS
from trainer.study import Study
from trainer.app import app_utils

os.environ['PROJECT_DIR'] = "/project"
os.environ['LOG_DIR'] = "/project/logdir"
os.environ['SSH_DIR'] = "/home/sandesh/studies"
os.environ['SSH_HOST'] = "10.19.137.42"
os.environ['SSH_USERNAME'] = "sandesh"
os.environ['SSH_PASSWORD'] = "letmein"
os.environ['MYSQL_HOST'] = "10.19.137.42"
os.environ['MYSQL_USERNAME'] = "sandesh"
os.environ['MYSQL_PASSWORD'] = "letmein"

if __name__ == "__main__":
    st.set_page_config(layout="wide") 
    with st.sidebar:
        with st.form(key='config_form') as user_config_form:
            study_cfg =  app_utils.load_config()

    if study_cfg is not None:

        study = Study(study_cfg)
        runs = study.show_runs()

        # Title
        st.title(pretty_title(study.name))
        st.markdown(study.config['desc'])
        # Study Defaults
        with st.expander("Default Study Config", expanded=False):
            app_utils.study_defaults(study)

        st.header("Select Runs")

        with st.expander("Select Runs", expanded=True):
            runs = app_utils.run_selector(study)

        with st.sidebar:
            runs, group_cols = app_utils.group_runs(runs)
        
        from st_aggrid import AgGrid, GridOptionsBuilder, ColumnsAutoSizeMode, JsCode
        screen_width = streamlit_js_eval(js_expressions='window.innerWidth', key = 'SCR_WIDTH')
        # chart_width = container_width/2
        # chart_height = container_width/2
        if runs.shape[0] > 0:
            container_width = screen_width
            # Plot metrics
            st.header("Metrics")
            for metric_name, metric in study.metrics.items():
                    with st.expander(pretty_title(metric_name), expanded=True) as container:
                        chart_size = st.slider("Chart Size", min_value=0.1, max_value=1.0, value=0.3,
                                                key=f"chart_size_{metric_name}", format="")
                        chart_size = container_width*chart_size
                        # Get metric data
                        metric_data = study.metric_table(metric_name, limit=None)
                        metric_data = metric_data.merge(runs, 
                                                        on=['run_id', 'sweep', 'project'],
                                                        how='right')
                        metric_data = metric_data.dropna(subset=['eval_name'])
                        
                        # Average out data over the groups                        
                        metric_data = app_utils.avg_over_group(metric_data)


                        if metric_data.shape[0] > 0:
                            st.subheader(pretty_title(metric_name))

                            num_facets = metric_data['eval_name'].nunique()
                            num_cols = int(container_width//chart_size)

                            if metric_name == 'observation_videos':
                                btn1, btn2, btn3 = st.columns(3)
                                project_select = st.selectbox("Select Project", metric_data['project'].unique())
                                sweep_select = st.selectbox("Select Sweep", metric_data.query(f"project=='{project_select}'")['sweep'].unique())
                                run_select = st.selectbox("Select Run", metric_data.query(f"project=='{project_select}' and sweep=='{sweep_select}'")['run_id'].unique())
                                # st.write(metric_data)
                                evals = metric_data.query(f"project=='{project_select}' and sweep=='{sweep_select}' and run_id=='{run_select}'")
                                num_vids = evals.shape[0]
                                vid_cols = st.columns(num_vids)
                                for idv in range(num_vids):
                                    data_row = evals.iloc[idv]
                                    filepath = data_row['filepath']
                                    filepath = filepath.split(study.storage.dir)[1].lstrip('/')
                                    video = study.storage.load(filepath, filetype='bytesio')
                                    vid_cols[idv].video(data=video)
                                    vid_cols[idv].write(f"Eval Name: `{pretty_title(data_row['eval_name'])}`")
                                    # metric.plot()
                            else:

                                facet = {'name': 'eval_name', 
                                        'columns': num_cols
                                        }
                                # st.write(metric_data)
                                chart, chart_info = metric.plot(metric_data, facet=facet, 
                                                                show_legend=False,
                                                                chart_properties={'width': chart_size, 
                                                                                'height': chart_size,}
                                                                                )
                                cols = [*group_cols, 'group']
                                data = metric_data.loc[:, cols].drop_duplicates()
                                legend_table = app_utils.legend_table(chart_info['legend_colors'], 'group', data)
                                st.dataframe(legend_table, hide_index=True)
                                st.altair_chart(chart)



    