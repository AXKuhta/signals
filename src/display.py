from plotly.subplots import make_subplots
import webbrowser
import plotly

style = """
<style>
body {
	font-family: sans-serif;
}

.minmaxplot {
	break-inside: avoid;
	break-after: always;
}

.plotly_html > div {
	height: 40rem;
}

.header, .footer {
	padding: 1rem 0 1rem 5%;
	text-align: center;
}
</style>
"""

# Intentionally cut the precision down to f32 for reduced file size
def ensure_lowp(x):
	if hasattr(x, "dtype") and x.dtype == "float64":
		return x.astype("float32")

	return x

class minmaxplot():
	"""
	Wrapper around plotly to make multiple traces with error bands of matching color.

	Supports optional secondary y axis.
	"""

	def __init__(self, x_unit=None, secondary_y=False, planar=False, scientific=True):
		self.reset_color_cycler()
		self.secondary_y = secondary_y
		self.x_logscale = False
		self.y_logscale = False
		self.y_title = None
		self.x_title = None
		self.x_range = []
		self.y_range = []
		self.x_unit = x_unit
		self.planar = planar
		self.scientific = scientific
		self.header_html = ""
		self.footer_html = ""
		self.traces = []

	def header(self, html):
		self.header_html = html

	def footer(self, html):
		self.footer_html = html

	def xtitle(self, title):
		self.x_title = title

	def ytitle(self, title):
		self.y_title = title

	def xlogscale(self, enable=True):
		self.x_logscale = enable

	def ylogscale(self, enable=True):
		self.y_logscale = enable

	def xrange(self, range):
		self.x_range = range

	def yrange(self, range):
		self.y_range = range

	def reset_color_cycler(self):
		self.hsl_color_cycler = [] + plotly.colors.qualitative.Plotly + plotly.colors.qualitative.T10 + ["#FF0000"]*90

	def trace(self, time, signal, name=None, hidden=False, error_band=None, markers=None, dash=None, width=None, secondary=False):
		"""
		Trace a line

		x			time
		y			signal
		name			legend name
		hidden			initial visibility toggle
		error_band		tuple of (min, max) for error band polygon
		markers			force markers on or off, defaults to on for traces with less than 500 points
		dash			dashed style toggle
		width			line width for making dashed lines look pretty
		secondary		right hand side y axis toggle
		"""

		color = self.hsl_color_cycler.pop(0)
		r, g, b = plotly.colors.hex_to_rgb(color)

		if not isinstance(markers, bool):
			markers = len(time) < 500

		plot_signal = plotly.graph_objects.Scatter(
			x=ensure_lowp(time),
			y=ensure_lowp(signal),
			name=name,
			line=dict(color=color, dash=dash, width=width), # shape="spline", smoothing=1 slow
			visible="legendonly" if hidden else True,
			mode="lines+markers" if markers else "lines"
		)

		if error_band is None:
			plot_error = {}
		else:
			min, max = error_band

			time = list(time)
			min = list(min)
			max = list(max)

			poly_x = time + time[::-1]
			poly_y = max + min[::-1]

			plot_error = plotly.graph_objects.Scatter(
				x=ensure_lowp(poly_x),
				y=ensure_lowp(poly_y),
				fill='toself',
				fillcolor=f'rgba({r}, {g}, {b}, .2)',
				line=dict(width=0),
				#line=dict(color='rgba(255,255,255,0)'),
				hoverinfo="skip",
				name=f"{name} minmax"
			)

		assert self.secondary_y if secondary else 1, "Please enable secondary_y=True when creating minmaxplot()"

		self.traces.append({
			"line": plot_signal,
			"error_band": plot_error,
			"secondary": secondary
		})

	def fig(self):
		#fig = plotly.graph_objects.Figure(data=self.plots, secondary_y=True)

		fig = make_subplots(specs=[[{
			"secondary_y": self.secondary_y,
			"l": 0.05, # Left padding to make ylabel fit
			"b": 0.06, # Bottom padding to make xlabel fit with diagonal ticklabels
		}]])

		for trace in self.traces:
			fig.add_trace(trace["line"], secondary_y=trace["secondary"])
			fig.add_trace(trace["error_band"], secondary_y=trace["secondary"])

		fig.update_layout(
			template=plotly.io.templates["none"],
			xaxis_ticksuffix=self.x_unit,
			xaxis_minexponent=2,
			font=dict( size=24 ),
			legend=dict(
				orientation="h",
				yanchor="bottom",
				y=1.02,
				xanchor="right",
				x=1
			) if len(self.traces) < 9 else None
		)

		fig.update_layout(
			yaxis=dict(
				side="left",
			),
			yaxis2=dict(
				side="right",
				overlaying="y",
				tickmode="sync",
				tickformat=',d',
				range=[0, 100]
			),
		)

		fig.update_xaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='grey',
			gridcolor='grey',
			tickcolor='grey',
			title=self.x_title
		)
		fig.update_yaxes(
			mirror=True,
			ticks='outside',
			showline=True,
			linecolor='grey',
			gridcolor='grey',
			tickcolor='grey',
			title=self.y_title,
		)

		if self.scientific:
			fig.update_yaxes(
				exponentformat="power",
				showexponent="last"
			)

		if self.x_range:
			fig.update_layout(xaxis_range=self.x_range)

		if self.y_range:
			fig.update_layout(yaxis_range=self.y_range)

		if self.y_logscale:
			fig.update_yaxes(type="log")

		if self.x_logscale:
			fig.update_xaxes(type="log")

		if self.planar:
			fig.update_yaxes(scaleanchor="x")

		return fig

	def show(self, title=""):
		fig = self.fig()

		with open("minmaxplot.html", "w") as file:
			file.write("<!DOCTYPE html>")
			file.write(style)
			file.write("<main>")
			file.write(f"<h3>{title}</h3>" if title else "<h3>WAVEFORM</h3>")
			file.write( fig.to_html(full_html=False) )
			file.write("</main>")

		webbrowser.open("minmaxplot.html")

class page():
	def __init__(self, figs, title=None):
		self.title = title
		self.figs = figs

	def write_fig(self, file, fig):
		file.write(f"<div class=\"{fig.__class__.__name__}\">")
		file.write(f"<div class=\"header\">{fig.header_html}</div>" if fig.header_html else "")
		file.write(f"<div class=\"plotly_html\">")
		file.write( fig.fig().to_html(full_html=False) )
		file.write("</div>")
		file.write(f"<div class=\"footer\">{fig.footer_html}</div>" if fig.footer_html else "")
		file.write("</div>")

	def write_fig_set(self, file, figs):
		for fig in figs:
			self.write_fig(file, fig)

	def show(self):
		with open("result.html", "w", encoding="utf8") as file:
			file.write("<!DOCTYPE html>")
			file.write(style)
			file.write("<main>")

			if self.title:
				file.write(f"<h1>{self.title}</h1>")

			for entry in self.figs:
				if type(entry) is list:
					self.write_fig_set(file, entry)
				else:
					self.write_fig(file, entry)

			file.write("</main>")

		webbrowser.open("result.html")

def waveform(time, signal, title=None, error_band=None):
	"""
	Display real (i.e. dtype is float64, not complex64) waveform

	time = time series
	signal = signal series
	[optional] error_band = error band polygon (time + time[::-1], points_max + points_min)
	"""

	samplerate = time.shape[0]

	plot_signal = plotly.graph_objects.Scatter(
		x=ensure_lowp(time),
		y=ensure_lowp(signal)
	)

	if error_band is None:
		plot_error = {}
	else:
		poly_x, poly_y = error_band

		plot_error = plotly.graph_objects.Scatter(
			x=ensure_lowp(poly_x),
			y=ensure_lowp(poly_y),
			fill='toself',
			fillcolor='rgba(0,100,80,0.2)',
			line=dict(color='rgba(255,255,255,0)'),
			hoverinfo="skip",
			name="error band"
		)

	fig_signal = plotly.graph_objects.Figure(data=[plot_signal, plot_error])

	fig_signal.update_layout(
		#yaxis_range=[-1,1],
		xaxis_ticksuffix="s",
		xaxis_minexponent=2,
		#xaxis_range=[0, 1.2/1000]
	)

	# Display steps if the signal is sufficiently low res
	if samplerate <= 3840:
		fig_signal.update_traces(line_shape="hv")

	with open("waveform.html", "w") as file:
		file.write("<!DOCTYPE html>")
		file.write(style)
		file.write("<main>")
		file.write(f"<h3>{title}</h3>" if title else "<h3>WAVEFORM</h3>")
		file.write( to_html(fig, full_html=False) )
		file.write("</main>")

	webbrowser.open("waveform.html")



def signal_fft(time, signal, title=None):
	"""
	Display signal waveform, spectrum and amplitude

	time = time series
	signal = signal series
	"""

	import numpy as np

	samples = time.shape[0]
	samplerate = samples / time.max()
	amplitude = np.abs(signal)
	angle = np.angle(signal)

	spectrum_x = np.linspace(-samplerate/2, samplerate/2, samples)
	spectrum = np.fft.fftshift( np.fft.fft(signal) )
	spectrum_abs = np.abs(spectrum)
	spectrum_angle = np.angle(spectrum)

	plot_signal_real = plotly.graph_objects.Scatter(x=time, y=signal.real, name="Real")
	plot_signal_imag = plotly.graph_objects.Scatter(x=time, y=signal.imag, name="Imag", visible="legendonly")
	plot_signal_abs = plotly.graph_objects.Scatter(x=time, y=amplitude, name="Mag")
	plot_signal_angle = plotly.graph_objects.Scatter(x=time, y=angle, name="Angle")

	plot_spectrum_real = plotly.graph_objects.Scatter(x=spectrum_x, y=spectrum.real, name="Real", visible="legendonly")
	plot_spectrum_imag = plotly.graph_objects.Scatter(x=spectrum_x, y=spectrum.imag, name="Imag", visible="legendonly")
	plot_spectrum_abs = plotly.graph_objects.Scatter(x=spectrum_x, y=spectrum_abs, name="Mag")
	plot_spectrum_angle = plotly.graph_objects.Scatter(x=spectrum_x, y=spectrum_angle, name="Angle")

	fig_signal = plotly.graph_objects.Figure(data=[plot_signal_real, plot_signal_imag])
	fig_spectrum = plotly.graph_objects.Figure(data=[plot_spectrum_real, plot_spectrum_imag, plot_spectrum_abs, plot_spectrum_angle])
	fig_amplitude = plotly.graph_objects.Figure(data=[plot_signal_abs, plot_signal_angle])

	fig_signal.update_layout(
		yaxis_range=[-1,1],
		xaxis_ticksuffix="s",
		xaxis_minexponent=2
	)

	fig_amplitude.update_layout(
		yaxis_range=[-1,1],
		xaxis_ticksuffix="s",
		xaxis_minexponent=2
	)

	fig_spectrum.update_layout(
		xaxis_ticksuffix="Hz",
	)

	# Display steps if the signal is sufficiently low res
	if samples <= 3840:
		fig_signal.update_traces(line_shape="hv")

	with open("fft_plot.html", "w") as file:
		file.write("<!DOCTYPE html>")
		file.write(style)
		file.write("<main>")
		if title:
			file.write(f"<h1>{title}</h1>")
		file.write("<h3>WAVEFORM</h3>")
		file.write(fig_signal.to_html(full_html=False))
		file.write("<h3>SPECTRUM</h3>")
		file.write(fig_spectrum.to_html(full_html=False))
		file.write("<h3>STRENGTH</h3>")
		file.write(fig_amplitude.to_html(full_html=False))
		file.write("</main>")

	webbrowser.open("fft_plot.html")

