import streamlit as st
from trainer.utils import import_module_attr, pretty_title

def setup_model_wizard():
    
    if 'model' not in st.session_state:
        st.session_state['model'] = {}

    st.header("Setup Model")
    with st.container(border=True):
        st.markdown("Give your model a name")
        model_name = st.text_input("Model Name", key="model_name", value=None)

    if model_name is not None:
        st.subheader(f"Setting up Model ``{pretty_title(model_name)}``")
        with st.container(border=True):
            st.markdown("""We now need the path to the model's module path so we can import it.
                        e.g. If you'd usually do ``from src.models import MyModel``,
                        you'll want to use the module_path ``src.models.my_model.ModelName``
                        """)
            module_path = st.text_input("Model Module Path", key="model_module_path")

            # Try to import it
            try_import = st.button("Try Import", key="try_import")
            from trainer.examples.supervised_study.ml_trainer import SimpleMLModel
            if try_import:
                try:
                    obj = import_module_attr(module_path)
                    st.success("✅ Module imported successfully")
                    model_import_success = True
                except Exception as e:
                    st.error("Couldn't import module. Error: " + str(e))
                    



        st.text_input("Agent Module Path", key="agent_module_path")