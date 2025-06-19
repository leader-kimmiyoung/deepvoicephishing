package com.quicinc.chatapp

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import android.graphics.Color

class CallListAdapter(
    private val callList: List<CallWithChunks>
) : RecyclerView.Adapter<CallListAdapter.CallViewHolder>() {

    inner class CallViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val tvCaller: TextView = view.findViewById(R.id.tvCaller)
        val tvTime: TextView = view.findViewById(R.id.tvTime)
        val chunkLayout: LinearLayout = view.findViewById(R.id.chunkLayout)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): CallViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_call, parent, false)
        return CallViewHolder(view)
    }

    override fun onBindViewHolder(holder: CallViewHolder, position: Int) {
        val item = callList[position]
        holder.tvCaller.text = item.call.caller

        holder.itemView.setOnClickListener {
            item.isExpanded = !item.isExpanded
            notifyItemChanged(position)
        }

        holder.chunkLayout.removeAllViews()
        if (item.isExpanded) {
            holder.chunkLayout.visibility = View.VISIBLE
            item.chunkResults.forEach { chunk ->
                val chunkView = TextView(holder.itemView.context).apply {
                    text = "청크 ${chunk.chunkNumber}: 딥보이스=${chunk.isDeepfake}, 피싱=${chunk.isPhishing}\n→ ${chunk.reason}"
                    setPadding(0, 4, 0, 4)
                    setTextColor(
                        if (chunk.isPhishing == "fake") Color.RED else Color.DKGRAY
                    )
                }
                holder.chunkLayout.addView(chunkView)
            }
        } else {
            holder.chunkLayout.visibility = View.GONE
        }
    }

    override fun getItemCount() = callList.size
}
