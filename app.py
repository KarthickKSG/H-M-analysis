fig_line = px.line(
        df_raw.sort_values(sort_col), 
        x=sort_col, y=f_z, color='Cluster',
        line_shape='spline', # Makes the line smooth
        color_discrete_sequence=palette_map[palette],
        template="plotly_dark"
    )
    fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_line, use_container_width=True)

with tab2:
    st.markdown("### Population Hierarchy")
    # Sunburst chart showing how data is nested
    path_cols = ['Cluster']
    if 'Country' in df_raw.columns: path_cols.append('Country')
    if 'Age' in df_raw.columns: path_cols.append('Age')
    
    fig_sun = px.sunburst(
        df_raw, path=path_cols, values=f_z,
        color='Cluster',
        color_discrete_sequence=palette_map[palette],
        template="plotly_dark"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

with tab3:
    st.markdown("### Cluster Value Distribution")
    fig_violin = px.violin(
        df_raw, y=f_z, x='Cluster', color='Cluster',
        box=True, points="all",
        color_discrete_sequence=palette_map[palette],
        template="plotly_dark"
    )
    st.plotly_chart(fig_violin, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
