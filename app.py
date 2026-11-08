"""Hanuri battery network decision dashboard. Run: streamlit run app.py"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import linprog
import pydeck as pdk

st.set_page_config(page_title='Battery Supply Chain | Board Simulator',page_icon='🔋',layout='wide')
st.markdown("""<style>
.stApp {background: #f5f8fd; color: #172b4d;}
[data-testid="stMetric"] {background: #ffffff; border: 1px solid #d9e4f1; border-radius: 12px; padding: 14px;}
h1,h2,h3 {color: #17365d !important;}
</style>""", unsafe_allow_html=True)
st.title('🌐 HANNURI | Global Battery Supply Chain Intelligence')
st.caption('INU GLR-2026-BAT · 2026–2045 · Case-study assumptions, not enacted EU legal thresholds')
ROOT=Path(__file__).parent
@st.cache_data
def load():
 p=ROOT/'source_data.xlsx'
 return {s:pd.read_excel(p,sheet_name=s) for s in ['Demand_Nodes','Factories','Factory_Demand_Lanes','Annual_Assumptions']}
d=load(); nodes=d['Demand_Nodes']; fac=d['Factories']; lanes=d['Factory_Demand_Lanes']; assumptions=d['Annual_Assumptions']
base_ids=fac.loc[fac.status=='Existing','factory_id'].tolist()
candidates=fac.loc[fac.status!='Existing',['factory_id','factory_name']]
with st.sidebar:
 st.header('Scenario settings')
 year=st.slider('Analysis year',2026,2045,2035)
 options={'No new factory':None,**dict(zip(candidates.factory_name,candidates.factory_id))}
 choice=st.selectbox('European investment alternative',list(options))
 method=st.radio('Carbon accounting',['Market-based','Location-based'],horizontal=True)
 enforce=st.checkbox('Enforce EU carbon footprint cap',value=False)
 cap=st.number_input('EU delivered footprint cap (kgCO₂e/kWh)',value=65.0 if year<2035 else 60.0,step=1.0)
 coverage=st.slider('Maximum customer supply share',0.1,1.0,0.7,0.05)
 penalty=st.slider('Unserved target penalty ($/kWh)',0,150,150,5)
 st.caption('New plants assumed fully operational in 2031+ for this static annual allocation; before 2031 they are excluded. This does not calculate project NPV.')

names=nodes.demand_id.tolist(); selected=base_ids+([options[choice]] if options[choice] and year>=2031 else [])
active=fac.set_index('factory_id').loc[selected].copy()
active['capacity']=active['target_capacity_gwh_yr']
if year==2026:active['capacity']=active['current_capacity_gwh_yr']
dem=nodes.set_index('demand_id').loc[names].copy()
dem['demand_gwh']=dem.annual_ev_battery_demand_GWh_2026*(1+dem.growth_rate_2026_2030)**(min(year,2030)-2026)*(1+dem.growth_rate_2031_2045)**max(0,year-2030)
dem['target_gwh']=dem.demand_gwh*coverage
m=len(active);n=len(dem)
ln=lanes.set_index(['factory_id','demand_id'])
cost=np.zeros((m,n)); footprint=np.zeros((m,n)); allowed=np.ones((m,n),dtype=bool)
for i,(fid,f) in enumerate(active.iterrows()):
 ef=f.grid_intensity_kgco2_per_kwh; ren=f.renewable_share
 embedded=45.6+38.8*ef*(1-ren if method=='Market-based' else 1)
 material=50/f.yield_rate
 electricity=38.8*f.electricity_price_usd_per_kwh
 for j,nid in enumerate(names):
  l=ln.loc[(fid,nid)]
  # Approximate delivered variable cost, year-2026 materials decline, no fixed OPEX
  decline=(0.985**(min(year,2030)-2026))*(0.995**max(0,year-2030))
  production=material*decline+f.conversion_cost_usd_per_kwh_ex_material*(0.99**min(year-2026,9))+electricity
  freight=float(l.freight_cost_usd_per_kwh_finished_pack)
  tariff=(production+freight)*float(l.tariff_pct_of_customs_value)/100
  cost[i,j]=production+freight+tariff
  footprint[i,j]=embedded+float(l.transport_co2e_kg_per_kwh_finished_pack)
  if enforce and nid.startswith('D_EU_') and footprint[i,j]>cap:allowed[i,j]=False

# Decision variables: x_ij shipment, u_j unserved share. Objective costs plus shortage penalty.
objective=np.r_[cost.ravel(),np.full(n,float(penalty))]
Aeq=np.zeros((n,m*n+n)); beq=dem.target_gwh.to_numpy(dtype=float)
for j in range(n):
 for i in range(m):Aeq[j,i*n+j]=1
 Aeq[j,m*n+j]=1
Aub=np.zeros((m,m*n+n));bub=active.capacity.to_numpy(dtype=float)
for i in range(m):Aub[i,i*n:(i+1)*n]=1
bounds=[(0,None) if allowed[i,j] else (0,0) for i in range(m) for j in range(n)]+[(0,None)]*n
sol=linprog(objective,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method='highs')
if not sol.success:st.error('Optimization failed: '+sol.message);st.stop()
x=sol.x[:m*n].reshape(m,n);unmet=sol.x[m*n:]
served=x.sum();target=beq.sum();europe_mask=np.array([s.startswith('D_EU_') for s in names])
col=st.columns(4)
col[0].metric('Target demand',f'{target:,.1f} GWh')
col[1].metric('Allocated',f'{served:,.1f} GWh')
col[2].metric('Unserved target',f'{unmet.sum():,.1f} GWh')
col[3].metric('EU allocated',f'{x[:,europe_mask].sum():,.1f} GWh')
st.info('Optimization is a one-year cost + unserved-target-penalty model. Penalty is a user-defined assumption; results are not a 20-year investment NPV or a confirmed OEM contract.')

tabs=st.tabs(['Executive Dashboard','🌍 Global Network Map','Allocation','Factories','Carbon','Investment scenarios','Risk Monitoring','Board Report & Evidence','Model notes'])
with tabs[1]:
 st.subheader(f'Global production & customer network · {year}')
 st.caption('Thin straight lines represent optimized factory-to-customer allocations (GWh), not actual shipping routes. Hover over lines for shipment amounts.')
 a,b,c=st.columns([1,1,1])
 with a: show_lanes=st.checkbox('Show allocated connections',value=True)
 with b: show_customers=st.checkbox('Show customer sites',value=True)
 with c: show_candidates=st.checkbox('Show candidate plants',value=True)
 factories_map=fac.copy()
 factories_map['is_active']=factories_map.factory_id.isin(selected)
 factories_map=factories_map[(factories_map.status=='Existing') | (show_candidates & (factories_map.status!='Existing'))].copy()
 factories_map['color']=factories_map.apply(lambda r: [0,142,112,240] if r['is_active'] else [234,137,30,225],axis=1)
 factories_map['radius']=factories_map.apply(lambda r: 65000 if r['is_active'] else 45000,axis=1)
 factories_map['label']=factories_map.apply(lambda r: ('ACTIVE' if r['is_active'] else 'CANDIDATE')+' | '+str(r.factory_name),axis=1)
 demand_map=dem.reset_index().copy()
 demand_map['allocated_gwh']=x.sum(axis=0)
 demand_map['color']=[[30,94,208,230]]*len(demand_map)
 demand_map['radius']=demand_map.allocated_gwh.apply(lambda v: 30000+min(float(v),20)*3500)
 arc_rows=[]
 for i,(fid,f) in enumerate(active.iterrows()):
  for j,nid in enumerate(names):
   amount=float(x[i,j])
   if amount>3.0:
    dest=dem.loc[nid]
    arc_rows.append({'source_name':str(f.factory_name),'target_name':str(dest.oem_site),
       'source_lon':float(f.longitude),'source_lat':float(f.latitude),
       'target_lon':float(dest.longitude),'target_lat':float(dest.latitude),
       'allocated_gwh':round(amount,3),'width':max(1.0,min(12.0,amount*1.2)),
       'source_color':[0,130,110,225],'target_color':[30,94,208,230]})
 arc_df=pd.DataFrame(arc_rows)
 layers=[]
 if show_lanes and len(arc_df):
  layers.append(pdk.Layer('LineLayer', arc_df, get_source_position='[source_lon, source_lat]', get_target_position='[target_lon, target_lat]', get_color=[35, 105, 180, 70], get_width=1, width_units='pixels', pickable=True))
 if show_customers:
  layers.append(pdk.Layer('ScatterplotLayer',demand_map,get_position='[longitude, latitude]',get_fill_color='color',get_radius='radius',pickable=True,auto_highlight=True))
 layers.append(pdk.Layer('ScatterplotLayer',factories_map,get_position='[longitude, latitude]',get_fill_color='color',get_radius='radius',pickable=True,auto_highlight=True,stroked=True,get_line_color=[30,55,80,200],line_width_min_pixels=1))
 view=pdk.ViewState(latitude=43,longitude=55,zoom=2.1,pitch=0)
 deck=pdk.Deck(layers=layers,initial_view_state=view,map_style='https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json',tooltip={'html':'<b>{factory_name}{oem_site}{source_name}</b><br/>{target_name}<br/>Allocated: {allocated_gwh} GWh<br/>{label}','style':{'backgroundColor':'#ffffff','color':'#17365d'}})
 st.pydeck_chart(deck,width='stretch',height=570)
 st.caption('🟢 Active factories   🟠 Candidate factories   🔵 Customer sites   🟢→🔵 Thin allocation lines (shipments > 3 GWh). Candidate positions are from the supplied workbook.')
 left,right=st.columns(2)
 with left:
  st.markdown('**Active production capacity**')
  st.dataframe(active[['factory_name','capacity']].rename(columns={'capacity':'GWh/year'}),width='stretch')
 with right:
  st.markdown('**Largest allocated customer lanes**')
  if len(arc_df):
   st.dataframe(arc_df[['source_name','target_name','allocated_gwh']].sort_values('allocated_gwh',ascending=False).head(12),width='stretch',hide_index=True)
  else: st.info('No shipments selected under current scenario. Try increasing shortage penalty or changing constraints.')
 st.download_button('Download mapped allocation CSV',arc_df.to_csv(index=False).encode('utf-8-sig'),'mapped_lanes.csv','text/csv')
with tabs[2]:
 out=dem[['oem_site','customer_group','demand_gwh','target_gwh']].copy()
 for i,(fid,f) in enumerate(active.iterrows()):out[f.factory_name]=x[i]
 out['allocated_gwh']=x.sum(axis=0);out['unserved_target_gwh']=unmet
 out['share_of_OEM_demand_pct']=100*out.allocated_gwh/out.demand_gwh
 st.subheader(f'OEM supply allocation | {year}')
 st.dataframe(out.reset_index(),width="stretch",hide_index=True)
 st.download_button('Download allocation CSV',out.to_csv().encode('utf-8-sig'),'allocation.csv','text/csv')
 st.bar_chart(out.set_index('oem_site')[['allocated_gwh','unserved_target_gwh']])
with tabs[3]:
 fr=active[['factory_name','status','capacity']].copy();fr['allocated_gwh']=x.sum(axis=1);fr['utilization_pct']=100*fr.allocated_gwh/fr.capacity
 st.dataframe(fr,width="stretch");st.bar_chart(fr.set_index('factory_name')[['capacity','allocated_gwh']])
with tabs[4]:
 carbon=[]
 for i,(fid,f) in enumerate(active.iterrows()):
  carbon.append({'factory':f.factory_name,'market_cf':45.6+38.8*f.grid_intensity_kgco2_per_kwh*(1-f.renewable_share),'location_cf':45.6+38.8*f.grid_intensity_kgco2_per_kwh,'EU_eligible_lanes':int(allowed[i,europe_mask].sum()),'EU_lanes':int(europe_mask.sum())})
 st.dataframe(pd.DataFrame(carbon),width="stretch",hide_index=True)
 st.caption('The eligibility rule applies only to EU-bound lanes; other destinations remain unrestricted by this EU scenario.')
with tabs[5]:
 scenarios=['Baseline','High carbon price','Footprint cap','Demand slowdown','EU local rule']
 matrix={'Poland extension':[-365,-159,-3680,-1365,-215],'Hungary': [263,523,2336,-1206,456],'Germany':[-2083,-1908,-166,-3367,-1902],'France':[-997,-706,521,-2065,-833],'Spain':[-167,78,1311,-1195,-9],'Sweden':[-2046,-1777,-608,-3053,-1907],'Portugal':[-798,-598,305,-1473,-680],'Czechia':[-511,-426,907,-1357,-375]}
 npv=pd.DataFrame(matrix,index=scenarios).T
 st.subheader('Network incremental NPV · Case exhibit 14 · USD million')
 st.dataframe(npv.style.format('{:+,.0f}'),width="stretch")
 probs=st.columns(5);p=[]
 for c,name in zip(probs,scenarios):p.append(c.number_input(name+' probability',min_value=0.0,max_value=1.0,value=0.2,step=0.05,key=name))
 if sum(p)>0:
  expectation=npv.to_numpy()@ (np.array(p)/sum(p))
  regret=(npv.max(axis=0)-npv).max(axis=1)
  st.dataframe(pd.DataFrame({'Expected incremental NPV (USD m)':expectation,'Maximum regret (USD m)':regret},index=npv.index).sort_values('Expected incremental NPV (USD m)',ascending=False),width="stretch")
 st.caption('Scenario NPV values are supplied by the case. This tab does not recompute them from the annual allocation solver.')
with tabs[8]:
 st.markdown('''**Optimization:** minimize sum of delivered variable cost × GWh shipped + shortage penalty × GWh unserved. Each customer allocation + unserved equals its maximum capturable demand. Each factory shipment is bounded by capacity. Optional EU footprint cap forbids noncompliant EU-bound routes. All decisions are nonnegative.\n\n**Important limitations:** Candidate plants treated as full capacity only from 2031; annual cash flows, tax, subsidies, ramp-up, working capital, cannibalization and investment NPV require a separate financial model. Delivered costs are a transparent approximation, not a reproduction of all case exhibit 11 items. Carbon accounting uses simplified case assumptions. This dashboard is a research prototype, not an operational planning system.''')


with tabs[0]:
 st.subheader(f'Executive Dashboard | {year}')
 st.markdown('**Board decision snapshot**')
 k1,k2,k3,k4=st.columns(4)
 k1.metric('Service rate',f'{100*served/target:.1f}%' if target else 'N/A')
 k2.metric('Active capacity',f'{active.capacity.sum():,.1f} GWh')
 k3.metric('Unserved target',f'{unmet.sum():,.1f} GWh')
 k4.metric('EU deliveries',f'{x[:,europe_mask].sum():,.1f} GWh')
 st.warning('These are scenario-model outputs, not confirmed orders or an investment NPV. Compare investment alternatives and carbon rules before recommending a board decision.')
 st.markdown('**Active production sites and utilization**')
 overview=active[['factory_name','capacity']].copy()
 overview['allocated_gwh']=x.sum(axis=1)
 overview['utilization_pct']=np.where(overview.capacity>0,100*overview.allocated_gwh/overview.capacity,0)
 st.dataframe(overview,width='stretch')
 st.bar_chart(overview.set_index('factory_name')[['capacity','allocated_gwh']])
 st.markdown('**How to use this dashboard**')
 st.write('Select an investment alternative, year, carbon method and shortage penalty in the sidebar. Review investment scenarios separately: the NPV matrix is taken from the case exhibit and is not recalculated by the annual allocation optimizer.')

with tabs[6]:
 st.subheader(f'Risk Monitoring & Decision Triggers | {year}')
 utilization=100*served/active.capacity.sum() if active.capacity.sum()>0 else 0
 service=100*served/target if target>0 else 0
 eu_target=float(dem.target_gwh.to_numpy()[europe_mask].sum())
 eu_service=100*x[:,europe_mask].sum()/eu_target if eu_target>0 else 0
 risk=pd.DataFrame([
  {'KPI':'Target service rate','Value':f'{service:.1f}%','Illustrative trigger':'Below 90%','Suggested board response':'Review capacity expansion and customer allocation'},
  {'KPI':'Factory utilization','Value':f'{utilization:.1f}%','Illustrative trigger':'Above 90%','Suggested board response':'Assess bottlenecks, ramp-up and backup capacity'},
  {'KPI':'Unserved target','Value':f'{unmet.sum():.1f} GWh','Illustrative trigger':'Above 0 GWh','Suggested board response':'Evaluate investment alternatives and service commitments'},
  {'KPI':'EU target service rate','Value':f'{eu_service:.1f}%','Illustrative trigger':'Below 90%','Suggested board response':'Prioritize EU sourcing and regulatory readiness'},
  {'KPI':'Carbon cap enforcement','Value':'ON' if enforce else 'OFF','Illustrative trigger':'EU carbon rule or accounting method changes','Suggested board response':'Recalculate eligibility under both carbon accounting methods'},
 ])
 st.dataframe(risk,width='stretch',hide_index=True)
 st.caption('Thresholds are illustrative management assumptions, not thresholds prescribed by the case or enacted law. They require board validation.')

with tabs[7]:
 st.subheader(f'Board Report & Evidence | {year}')
 st.caption('Evidence for the Korean PDF board presentation. Values depend on the selected scenario.')
 group=out.groupby('customer_group',dropna=False)[['demand_gwh','target_gwh','allocated_gwh','unserved_target_gwh']].sum().reset_index()
 group['share_of_demand_pct']=np.where(group.demand_gwh>0,100*group.allocated_gwh/group.demand_gwh,0)
 st.markdown('**Customer-group supply allocation**')
 st.dataframe(group,width='stretch',hide_index=True)
 st.download_button('Download customer-group evidence CSV',group.to_csv(index=False).encode('utf-8-sig'),file_name='customer_group_evidence.csv',mime='text/csv')
 report=(f'HANNURI | BOARD EVIDENCE\nAnalysis year: {year}\nInvestment alternative: {choice}\n'
         f'Carbon accounting: {method}\nCarbon cap enforced: {enforce}\n'
         f'Maximum customer supply share: {coverage:.0%}\nUnserved penalty: ${penalty}/kWh\n'
         f'Target demand: {target:.2f} GWh\nAllocated: {served:.2f} GWh\n'
         f'Unserved target: {unmet.sum():.2f} GWh\nEU allocated: {x[:,europe_mask].sum():.2f} GWh\n'
         'MODEL LIMITATION: Annual cost allocation is not investment NPV. NPV values are supplied separately by the case exhibit.\n')
 st.download_button('Download board evidence TXT',report.encode('utf-8'),file_name='board_evidence.txt',mime='text/plain')
 st.info('For the final PDF, cite the case exhibits, include scenario assumptions, and provide the publicly deployed app URL only after testing it in a private browser window.')
