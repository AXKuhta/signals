from plotly.subplots import make_subplots
import webbrowser
import plotly
import torch

from .plotly_custom_html import to_html

style = """
<style>
body {
	font-family: sans-serif;
}

main > * {
	border-left: .5rem solid #1abc9c;
}

main > h3 {
	border-color: #2c3e50;
}

main > div {
	height: 40rem;
	break-inside: avoid;
	break-after: always;
}

h1, h3 {
	padding: 0 1rem;
	margin: 0;
}
</style>
"""

class minmaxplot():
	"""
	Wrapper around plotly to make multiple traces with error bands of matching color.

	Supports optional secondary y axis.
	"""

	def __init__(self, x_unit=None, title="None", logscale=False, secondary_y=False, planar=False):
		self.reset_color_cycler()
		self.secondary_y = secondary_y
		self.logscale = logscale
		self.y_title = None
		self.x_title = None
		self.x_unit = x_unit
		self.planar = planar
		self.title = title
		self.traces = []

	def xtitle(self, title):
		self.x_title = title

	def ytitle(self, title):
		self.y_title = title

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
			x=time,
			y=signal,
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
				x=poly_x,
				y=poly_y,
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
			"l": .05 # Left padding to make ylabel fit
		}]])

		for trace in self.traces:
			fig.add_trace(trace["line"], secondary_y=trace["secondary"])
			fig.add_trace(trace["error_band"], secondary_y=trace["secondary"])

		fig.update_layout(
			template=plotly.io.templates["none"],
			xaxis_ticksuffix=self.x_unit,
			xaxis_minexponent=2,
			#xaxis_range=[0, 1.2/1000],
			#yaxis_range=[-1,1],
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
			title=self.y_title
		)

		if self.logscale:
			fig.update_yaxes(type="log")

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
			file.write( to_html(fig, full_html=False) )
			file.write("</main>")

		webbrowser.open("minmaxplot.html")

class page():
	def __init__(self, figs, title=None):
		self.title = title
		self.figs = figs

	def write_fig(self, file, fig):
		if fig.title:
			file.write(f"<h3>{fig.title}</h3>")

		file.write( to_html(fig.fig(), full_html=False) )

	def write_fig_set(self, file, figs):
		for fig in figs:
			self.write_fig(file, fig)

	def show(self):
		with open("result.html", "w") as file:
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

	plot_signal = plotly.graph_objects.Scatter(x=time, y=signal)

	if error_band is None:
		plot_error = {}
	else:
		poly_x, poly_y = error_band

		plot_error = plotly.graph_objects.Scatter(
			x=poly_x,
			y=poly_y,
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

	samples = time.shape[0]
	samplerate = samples / time.max()
	amplitude = signal.abs()
	angle = signal.angle()

	spectrum_x = torch.linspace(-samplerate/2, samplerate/2, samples)
	spectrum = torch.fft.fftshift( torch.fft.fft(signal) )
	spectrum_abs = spectrum.abs()
	spectrum_angle = spectrum.angle()

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
		file.write(to_html(fig_signal, full_html=False))
		file.write("<h3>SPECTRUM</h3>")
		file.write(to_html(fig_spectrum, full_html=False))
		file.write("<h3>STRENGTH</h3>")
		file.write(to_html(fig_amplitude, full_html=False))
		file.write("</main>")

	webbrowser.open("fft_plot.html")

