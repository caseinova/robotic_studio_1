const sourceRoot = 'https://github.com/caseinova/robotic_studio_1/blob/project_page/';
document.querySelectorAll('[data-src]').forEach(a => { a.href = sourceRoot + a.dataset.src; });
async function configTable(file, target, rows) {
  try {
    const response = await fetch('data/' + file);
    if (!response.ok) throw new Error('Configuration unavailable');
    const data = await response.json();
    const table = document.createElement('table');
    const caption = table.createCaption(); caption.textContent = 'Checked-in values · ' + file;
    const head = table.createTHead().insertRow();
    ['Parameter', 'Value', 'Meaning / units'].forEach(text => { const th = document.createElement('th'); th.textContent = text; head.append(th); });
    const body = table.createTBody();
    rows.forEach(([key, label, unit]) => {
      const value = key.split('.').reduce((o, k) => o?.[k], data);
      const row = body.insertRow();
      [label, value === undefined ? 'Not configured' : String(value), unit].forEach(text => { row.insertCell().textContent = text; });
    });
    document.getElementById(target).replaceChildren(table);
  } catch {
    const p = document.createElement('p'); p.textContent = 'Configuration table unavailable. See the linked JSON source in the repository.';
    document.getElementById(target).replaceChildren(p);
  }
}
configTable('controller_config.json', 'controller-table', [
  ['mode', 'Controller mode', 'Selected approach / transfer policy'],
  ['tool_pose_source', 'Tool feedback', 'Visual pose with FK fallback'],
  ['approach_linear_speed_m_s', 'Approach speed', 'm/s'],
  ['approach_angular_speed_rad_s', 'Approach angular speed', 'rad/s'],
  ['pickup_linear_speed_m_s', 'Pickup speed', 'm/s'],
  ['tool_clearance', 'Tool standoff', 'm; not collision clearance'],
  ['grasp_height_offset_m', 'Grasp offset', 'm'],
  ['tool_yaw_deg', 'Desired yaw', 'degrees']
]);
configTable('estimator_config.json', 'estimator-table', [
  ['cube_enabled', 'Cube EKF enabled', 'Boolean'],
  ['tool_enabled', 'Tool EKF enabled', 'Boolean'],
  ['cube.position_measurement_std', 'Cube position measurement σ', 'm'],
  ['cube.orientation_measurement_std', 'Cube orientation measurement σ', 'rad'],
  ['cube.linear_acceleration_noise', 'Cube linear process-noise scale', 'See EKF discretization'],
  ['cube.angular_acceleration_noise', 'Cube angular process-noise scale', 'See EKF discretization'],
  ['cube.innovation_gate', 'Innovation gate', 'Squared Mahalanobis threshold'],
  ['tool.prediction_model', 'Tool prediction', 'FK increments'],
  ['tool.fk_position_noise', 'Tool FK position process noise', 'm/√s'],
  ['tool.fk_orientation_noise', 'Tool FK orientation process noise', 'rad/√s'],
  ['tool.max_prediction_age', 'Tool prediction age limit', 's']
]);
